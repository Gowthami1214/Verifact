"""
scripts/train_claim_extractor.py
----------------------------------
Fine-tunes a DeBERTa-v3-small sentence classifier on Layer 3 claim extraction data.

Training data: dataset/layer3_claim_extraction/claim_spans.jsonl
  - Each line: {doc_id, sentence_id, sentence_text, is_factual_claim, claim_type, claim_span}
  - Label: is_factual_claim (bool -> 0/1)

Model: microsoft/deberta-v3-small  (fast, accurate, good for sentence classification)
Output: models/claim_extraction/deberta-v3-small-claim-clf/

Usage:
    python scripts/train_claim_extractor.py [--epochs 5] [--batch-size 16] [--lr 2e-5]
"""

import os, sys, json, argparse, random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Config ───────────────────────────────────────────────────────────────────
CLAIM_SPANS_PATH = os.path.join(ROOT, "dataset", "layer3_claim_extraction", "claim_spans.jsonl")
TEST_SPANS_PATH  = os.path.join(ROOT, "dataset", "layer3_claim_extraction", "claim_extraction_test.jsonl")
MODEL_OUTPUT_DIR = os.path.join(ROOT, "models", "claim_extraction", "deberta-v3-small-claim-clf")
BASE_MODEL       = "microsoft/deberta-v3-small"

LABEL_MAP   = {False: 0, True: 1}
ID2LABEL    = {0: "non_claim", 1: "factual_claim"}
LABEL2ID    = {"non_claim": 0, "factual_claim": 1}


# ── Dataset ──────────────────────────────────────────────────────────────────
class ClaimDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=128):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["sentence_text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(LABEL_MAP[rec["is_factual_claim"]], dtype=torch.long),
        }


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    acc       = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-len",    type=int,   default=128)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading training data from: {CLAIM_SPANS_PATH}")
    train_records = load_jsonl(CLAIM_SPANS_PATH)
    print(f"  Train samples: {len(train_records)}")
    train_labels  = Counter(r["is_factual_claim"] for r in train_records)
    print(f"  Label distribution: {dict(train_labels)}")

    print(f"\nLoading test data from: {TEST_SPANS_PATH}")
    test_records = load_jsonl(TEST_SPANS_PATH)
    print(f"  Test samples: {len(test_records)}")

    random.shuffle(train_records)

    # ── Tokenizer & Model ──────────────────────────────────────────────────────
    print(f"\nLoading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_ds = ClaimDataset(train_records, tokenizer, max_len=args.max_len)
    test_ds  = ClaimDataset(test_records,  tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 5 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {total_loss/step:.4f}")

        # Evaluate after each epoch
        metrics = evaluate(model, test_loader, device)
        print(f"\n  Epoch {epoch} Test Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
            print(f"  Best model saved (F1={best_f1:.4f}) -> {MODEL_OUTPUT_DIR}")

        print()

    print("=" * 60)
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
