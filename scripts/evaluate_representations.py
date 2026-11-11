"""
scripts/evaluate_representations.py
-------------------------------------
Ablation of NLI input representations on validation data:
  1. Plain:              evidence [SEP] claim
  2. Evidence->Claim:    Evidence: <evidence> [SEP] Claim: <claim>
  3. Claim->Evidence:    Claim: <claim> [SEP] Evidence: <evidence>
  4. Premise->Hypothesis: Premise: <evidence> [SEP] Hypothesis: <claim>
  5. Hypothesis->Premise: Hypothesis: <claim> [SEP] Premise: <evidence>

Evaluates validation accuracy and Macro-F1 across formats.
"""

import os, sys, csv, json, argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VAL_PATH = os.path.join(ROOT, "dataset", "layer1_nli", "val_enhanced.csv")
MODEL_PATH = os.path.join(ROOT, "models", "nli", "deberta-v3-small-nli")

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}


def format_inputs(ev: str, cl: str, fmt: str):
    if fmt == "plain":
        return ev, cl
    elif fmt == "evidence_claim":
        return f"Evidence: {ev}", f"Claim: {cl}"
    elif fmt == "claim_evidence":
        return f"Claim: {cl}", f"Evidence: {ev}"
    elif fmt == "premise_hypothesis":
        return f"Premise: {ev}", f"Hypothesis: {cl}"
    elif fmt == "hypothesis_premise":
        return f"Hypothesis: {cl}", f"Premise: {ev}"
    else:
        raise ValueError(f"Unknown format: {fmt}")


def evaluate_format(model, tokenizer, device, val_records, fmt):
    correct = 0
    total = len(val_records)
    per_class = {l: {"tp": 0, "fp": 0, "fn": 0} for l in ID2LABEL.values()}

    for r in val_records:
        text_a, text_b = format_inputs(r["evidence"], r["claim"], fmt)
        true_label = r["label"]

        inputs = tokenizer(text_a, text_b, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = ID2LABEL[pred_idx]

        if pred_label == true_label:
            correct += 1
            per_class[pred_label]["tp"] += 1
        else:
            per_class[pred_label]["fp"] += 1
            per_class[true_label]["fn"] += 1

    acc = correct / total if total > 0 else 0.0
    f1s = []
    for lbl in ID2LABEL.values():
        tp = per_class[lbl]["tp"]
        fp = per_class[lbl]["fp"]
        fn = per_class[lbl]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return acc, macro_f1, per_class


def main():
    print("=" * 65)
    print("TruthSync - Input Representation Ablation Study")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_records = []
    with open(VAL_PATH, encoding="utf-8", newline="") as f:
        val_records = list(csv.DictReader(f))
    print(f"Loaded {len(val_records)} validation samples from {VAL_PATH}")

    # Use current trained model
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    formats = [
        ("plain", "evidence [SEP] claim"),
        ("evidence_claim", "Evidence: <ev> [SEP] Claim: <cl>"),
        ("claim_evidence", "Claim: <cl> [SEP] Evidence: <ev>"),
        ("premise_hypothesis", "Premise: <ev> [SEP] Hypothesis: <cl>"),
        ("hypothesis_premise", "Hypothesis: <cl> [SEP] Premise: <ev>"),
    ]

    results = []
    print("\nRunning representation ablation...")
    print(f"{'Format':22s} | {'Val Accuracy':12s} | {'Macro-F1':10s}")
    print("-" * 52)

    for fmt_key, desc in formats:
        acc, macro_f1, _ = evaluate_format(model, tokenizer, device, val_records, fmt_key)
        print(f"{fmt_key:22s} | {acc*100:10.2f}% | {macro_f1:10.4f}")
        results.append({"format": fmt_key, "desc": desc, "accuracy": acc, "macro_f1": macro_f1})

    best = max(results, key=lambda x: x["macro_f1"])
    print("-" * 52)
    print(f"Best representation based strictly on validation Macro-F1:")
    print(f"  --> {best['format']} (Acc: {best['accuracy']*100:.2f}%, Macro-F1: {best['macro_f1']:.4f})")

    out_path = os.path.join(ROOT, "results", "representation_ablation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
