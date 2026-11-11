"""
scripts/train_nli_classifier.py
---------------------------------
Fine-tunes DeBERTa-v3-small on TruthSync Enhanced NLI dataset.

Key Enhancements:
  - Input representation: 'claim_evidence' (Claim: <claim> [SEP] Evidence: <evidence>)
    empirically verified as highest validation Macro-F1.
  - Dynamic padding via DataCollatorWithPadding for 2-3x faster CPU execution.
  - Stratified balanced training: train_enhanced.csv (612 rows) / val_enhanced.csv (105 rows).
  - Class-weighted CrossEntropyLoss to address class confusion.
  - Early stopping with patience=2 on validation Macro-F1.
  - Linear learning rate decay with 10% warmup.
  - Saves best model to models/deberta_best/ and tokenizer to models/tokenizer/.
"""

import os, sys, csv, json, argparse, random
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

TRAIN_PATH   = os.path.join(ROOT, "dataset", "layer1_nli", "train_enhanced.csv")
VAL_PATH     = os.path.join(ROOT, "dataset", "layer1_nli", "val_enhanced.csv")
MODEL_DIR    = os.path.join(ROOT, "models", "deberta_best")
TOKENIZER_DIR = os.path.join(ROOT, "models", "tokenizer")
BASE_MODEL   = "microsoft/deberta-v3-small"

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}
NUM_LABELS = 3


class DynamicNLIDataset(Dataset):
    def __init__(self, records, tokenizer, format_type="claim_evidence"):
        self.records = records
        self.tokenizer = tokenizer
        self.format_type = format_type

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        claim = r["claim"]
        evidence = r["evidence"]

        if self.format_type == "claim_evidence":
            text_a = f"Claim: {claim}"
            text_b = f"Evidence: {evidence}"
        else:
            text_a = evidence
            text_b = claim

        enc = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
        )
        enc["labels"] = LABEL2ID[r["label"]]
        return enc


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_class_weights(records, device):
    labels = [LABEL2ID[r["label"]] for r in records]
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (NUM_LABELS * counts[i]) for i in range(NUM_LABELS)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class = {l: {"tp": 0, "fp": 0, "fn": 0} for l in ID2LABEL.values()}
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, y in zip(preds.tolist(), labels.tolist()):
                pred_label = ID2LABEL[p]
                true_label = ID2LABEL[y]
                if p == y:
                    per_class[pred_label]["tp"] += 1
                else:
                    per_class[pred_label]["fp"] += 1
                    per_class[true_label]["fn"] += 1

    acc = correct / total if total > 0 else 0.0
    f1s, precisions, recalls = [], [], []
    report = {}

    for lbl in ID2LABEL.values():
        tp = per_class[lbl]["tp"]
        fp = per_class[lbl]["fp"]
        fn = per_class[lbl]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        precisions.append(p)
        recalls.append(r)
        report[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "refutes_f1": report["REFUTES"]["f1"],
        "per_class": report
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("TruthSync - Optimized DeBERTa-v3-small Training")
    print(f"  Device        : {device}")
    print(f"  Batch size    : {args.batch_size} (Dynamic Padding enabled)")
    print(f"  Learning Rate : {args.lr}")
    print(f"  Epochs        : {args.epochs} (Early Stopping patience={args.patience})")
    print("=" * 65)

    train_records = load_csv(TRAIN_PATH)
    val_records = load_csv(VAL_PATH)
    print(f"Loaded {len(train_records)} train rows, {len(val_records)} val rows.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_dataset = DynamicNLIDataset(train_records, tokenizer, format_type="claim_evidence")
    val_dataset = DynamicNLIDataset(val_records, tokenizer, format_type="claim_evidence")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    class_weights = compute_class_weights(train_records, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nStarting fine-tuning...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = round(train_loss / len(train_loader), 4)

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Macro-F1: {val_metrics['macro_f1']:.4f} | REFUTES F1: {val_metrics['refutes_f1']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(TOKENIZER_DIR, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(TOKENIZER_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  --> Saved new best checkpoint (Macro-F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at Epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    print("=" * 65)
    print(f"Training Complete! Best Checkpoint at Epoch {best_epoch}: Macro-F1 = {best_val_f1:.4f}")
    print(f"Saved to: {MODEL_DIR}")

    out_metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "hyperparameters": vars(args)
    }
    with open(os.path.join(ROOT, "results", "nli_training_optimized.json"), "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)


if __name__ == "__main__":
    main()
