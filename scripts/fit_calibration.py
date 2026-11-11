"""
scripts/fit_calibration.py
----------------------------
Fits Temperature Scaling parameter T on validation set logits using
src/nli/calibration.py, computes Expected Calibration Error (ECE) and
Brier score before and after calibration, and saves optimal temperature.
"""

import os, sys, csv, json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.nli.calibration import TemperatureScaler, compute_ece, compute_brier_score, get_reliability_data

VAL_PATH = os.path.join(ROOT, "dataset", "layer1_nli", "val_enhanced.csv")
MODEL_DIR = os.path.join(ROOT, "models", "deberta_best")
OUT_JSON = os.path.join(ROOT, "results", "calibration_results.json")

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}


def main():
    print("=" * 65)
    print("TruthSync - Confidence Calibration & Temperature Scaling")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_records = []
    with open(VAL_PATH, encoding="utf-8", newline="") as f:
        val_records = list(csv.DictReader(f))
    print(f"Loaded {len(val_records)} validation samples.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    logits_list = []
    labels_list = []

    for r in val_records:
        text_a = f"Claim: {r['claim']}"
        text_b = f"Evidence: {r['evidence']}"
        enc = tokenizer(text_a, text_b, max_length=256, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            l = model(**enc).logits
            logits_list.append(l.cpu())
            labels_list.append(LABEL2ID[r["label"]])

    logits = torch.cat(logits_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    # ── Before Calibration ────────────────────────────────────────────────────
    uncalibrated_probs = F.softmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()

    ece_before = compute_ece(uncalibrated_probs, labels_np)
    brier_before = compute_brier_score(uncalibrated_probs, labels_np)
    reliability_before = get_reliability_data(uncalibrated_probs, labels_np)

    print(f"\nUncalibrated Metrics:")
    print(f"  ECE (Expected Calibration Error) : {ece_before:.4f}")
    print(f"  Brier Score                      : {brier_before:.4f}")

    # ── Fit Temperature ───────────────────────────────────────────────────────
    scaler = TemperatureScaler()
    best_temp = scaler.fit(logits, labels)
    print(f"\nOptimized Temperature T = {best_temp:.4f}")

    # ── After Calibration ─────────────────────────────────────────────────────
    calibrated_logits = scaler(logits)
    calibrated_probs = F.softmax(calibrated_logits, dim=-1).detach().numpy()

    ece_after = compute_ece(calibrated_probs, labels_np)
    brier_after = compute_brier_score(calibrated_probs, labels_np)
    reliability_after = get_reliability_data(calibrated_probs, labels_np)

    print(f"\nCalibrated Metrics (T={best_temp:.4f}):")
    print(f"  ECE (Expected Calibration Error) : {ece_after:.4f} (Change: {ece_after - ece_before:+.4f})")
    print(f"  Brier Score                      : {brier_after:.4f} (Change: {brier_after - brier_before:+.4f})")

    results = {
        "optimal_temperature": round(best_temp, 4),
        "uncalibrated": {
            "ece": ece_before,
            "brier_score": brier_before,
            "reliability": reliability_before
        },
        "calibrated": {
            "ece": ece_after,
            "brier_score": brier_after,
            "reliability": reliability_after
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration results saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
