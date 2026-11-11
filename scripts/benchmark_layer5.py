"""
scripts/benchmark_layer5.py
-----------------------------
Full benchmark evaluation of TruthSync on the Layer 5 held-out test set.

Metrics:
  Claim-level:
    - Accuracy, Precision, Recall, Macro-F1 per class
    - Confusion matrix (SUPPORTS / REFUTES / NOT_ENOUGH_INFO)

  Response-level:
    - Verdict accuracy (exact match)
    - MAE on factuality_score
    - Weighted factuality score correlation (Pearson r)

Saves a full JSON report to models/benchmark_results.json and a
readable summary to models/benchmark_summary.txt.

Usage:
    python scripts/benchmark_layer5.py [--n 200] [--nli finetuned|pretrained]
"""

import os, sys, csv, json, argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

from src.pipeline import TruthSyncPipeline

RESPONSES_PATH  = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_responses.csv")
CLAIMS_PATH     = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_claims.csv")
OUT_JSON        = os.path.join(ROOT, "models", "benchmark_results.json")
OUT_SUMMARY     = os.path.join(ROOT, "models", "benchmark_summary.txt")

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den  = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den > 0 else 0.0


def compute_metrics(y_true, y_pred, labels):
    per_class = {l: {"tp": 0, "fp": 0, "fn": 0} for l in labels}
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
            per_class[pred]["tp"] += 1
        else:
            per_class[pred]["fp"] += 1
            per_class[true]["fn"] += 1

    accuracy = correct / len(y_true) if y_true else 0.0
    f1s, precisions, recalls = [], [], []
    per_class_report = {}
    for lbl in labels:
        tp = per_class[lbl]["tp"]
        fp = per_class[lbl]["fp"]
        fn = per_class[lbl]["fn"]
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        f1s.append(f1)
        precisions.append(p)
        recalls.append(r)
        per_class_report[lbl] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    macro_f1  = round(sum(f1s)        / len(labels), 4)
    macro_p   = round(sum(precisions)  / len(labels), 4)
    macro_r   = round(sum(recalls)     / len(labels), 4)

    # Confusion matrix
    conf = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        conf[true][pred] += 1

    return {
        "accuracy":  round(accuracy, 4),
        "macro_f1":  macro_f1,
        "macro_p":   macro_p,
        "macro_r":   macro_r,
        "per_class": per_class_report,
        "confusion_matrix": {t: dict(p_dict) for t, p_dict in conf.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=200,
                        help="Number of held-out responses to evaluate (default: 200)")
    parser.add_argument("--nli", default=None, choices=["pretrained", "finetuned"],
                        help="NLI mode override")
    args = parser.parse_args()

    print("=" * 65)
    print(f"TruthSync - Layer 5 Benchmark Evaluation")
    print(f"  N responses : {args.n}  |  NLI mode: {args.nli or 'auto'}")
    print("=" * 65)

    pipeline = TruthSyncPipeline(nli_mode=args.nli)

    responses  = load_csv(RESPONSES_PATH)[:args.n]
    claims_all = load_csv(CLAIMS_PATH)

    # Gold label index by response_id -> {claim_text: label}
    gt_claims_by_resp = defaultdict(dict)
    for c in claims_all:
        gt_claims_by_resp[c["response_id"]][c["claim_text"].strip()] = c["label"]

    # Run pipeline
    all_true, all_pred      = [], []
    score_true, score_pred  = [], []
    verdict_matches         = 0
    per_resp_results        = []

    for i, resp in enumerate(responses, 1):
        resp_id    = resp["response_id"]
        text       = resp["generated_text"]
        gt_verdict = resp["response_verdict"]
        gt_score   = float(resp["factuality_score"])

        report = pipeline.verify(text)

        for pred_claim in report["claims"]:
            gold = gt_claims_by_resp[resp_id].get(pred_claim["claim_text"].strip())
            if gold:
                all_true.append(gold)
                all_pred.append(pred_claim["label"])

        score_true.append(gt_score)
        score_pred.append(report["factuality_score"])
        if report["response_verdict"] == gt_verdict:
            verdict_matches += 1

        per_resp_results.append({
            "response_id":     resp_id,
            "pred_verdict":    report["response_verdict"],
            "gold_verdict":    gt_verdict,
            "pred_score":      report["factuality_score"],
            "gold_score":      gt_score,
            "n_claims":        report["total_claims"],
        })

        if i % 10 == 0 or i == len(responses):
            print(f"  [{i:3d}/{len(responses)}] {resp_id} "
                  f"pred={report['factuality_score']:.2f} '{report['response_verdict'][:18]}' "
                  f"gold={gt_score:.2f} '{gt_verdict[:18]}'")

    # ── Claim-level metrics ────────────────────────────────────────────────────
    claim_metrics = compute_metrics(all_true, all_pred, LABELS)

    # ── Response-level metrics ────────────────────────────────────────────────
    mae = round(sum(abs(p - t) for p, t in zip(score_pred, score_true)) / len(score_true), 4)
    r   = round(pearson_r(score_pred, score_true), 4)
    verdict_acc = round(verdict_matches / len(responses), 4)

    # ── Print summary ──────────────────────────────────────────────────────────
    label_dist = Counter(all_pred)
    summary_lines = [
        "=" * 65,
        "BENCHMARK RESULTS",
        "=" * 65,
        "",
        f"Evaluated:  {len(responses)} responses  |  {len(all_true)} matched claims",
        "",
        "CLAIM-LEVEL",
        f"  Accuracy       : {claim_metrics['accuracy']:.4f}",
        f"  Macro-F1       : {claim_metrics['macro_f1']:.4f}",
        f"  Macro-Precision: {claim_metrics['macro_p']:.4f}",
        f"  Macro-Recall   : {claim_metrics['macro_r']:.4f}",
        "",
        "  Per-class:",
    ]
    for lbl in LABELS:
        m = claim_metrics["per_class"][lbl]
        summary_lines.append(
            f"    {lbl:20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  support={m['support']}"
        )

    summary_lines += [
        "",
        "  Predicted label distribution:",
    ]
    for lbl in LABELS:
        summary_lines.append(f"    {lbl:20s}: {label_dist.get(lbl, 0)}")

    summary_lines += [
        "",
        "RESPONSE-LEVEL",
        f"  Verdict accuracy : {verdict_acc:.4f} ({verdict_matches}/{len(responses)})",
        f"  Score MAE        : {mae:.4f}",
        f"  Score Pearson r  : {r:.4f}",
        "=" * 65,
    ]

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    # ── Save ──────────────────────────────────────────────────────────────────
    full_results = {
        "n_responses":      len(responses),
        "n_matched_claims": len(all_true),
        "claim_metrics":    claim_metrics,
        "response_metrics": {
            "verdict_accuracy":  verdict_acc,
            "score_mae":         mae,
            "score_pearson_r":   r,
        },
        "per_response": per_resp_results,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nJSON results : {OUT_JSON}")
    print(f"Text summary : {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
