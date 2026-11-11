"""
scripts/demo_claim_extraction.py
----------------------------------
Demonstrates claim extraction on real AI-generated responses from
layer5_heldout/heldout_responses.csv using the rule-based extractor.

Shows:
  - Sentence-level annotations (is_factual_claim, claim_type, span)
  - Comparison with ground-truth labels from heldout_claims.csv

Usage:
    python scripts/demo_claim_extraction.py [--mode rule|nli] [--n 5]
"""

import os, sys, csv, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.claim_extraction.claim_extractor import ClaimExtractor

RESPONSES_PATH = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_responses.csv")
CLAIMS_PATH    = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_claims.csv")


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rule", choices=["rule", "nli"],
                        help="Extractor mode: 'rule' (fast) or 'nli' (zero-shot)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of responses to demo (default: 5)")
    args = parser.parse_args()

    extractor = ClaimExtractor(mode=args.mode)

    responses = load_csv(RESPONSES_PATH)[:args.n]
    claims_all = load_csv(CLAIMS_PATH)

    # Index ground-truth claims by response_id
    gt_by_resp = {}
    for c in claims_all:
        gt_by_resp.setdefault(c["response_id"], []).append(c)

    print("=" * 70)
    print(f"TruthSync - Claim Extraction Demo  (mode={args.mode})")
    print("=" * 70)

    total_predicted = 0
    total_gt        = 0
    total_correct   = 0

    for resp in responses:
        resp_id      = resp["response_id"]
        text         = resp["generated_text"]
        gt_claims    = gt_by_resp.get(resp_id, [])
        gt_texts_set = {c["claim_text"].strip().lower() for c in gt_claims}

        annotations  = extractor.extract(text)
        pred_claims  = [a for a in annotations if a["is_factual_claim"]]

        print(f"\n{'-'*70}")
        print(f"Response ID : {resp_id}  |  Topic: {resp['topic']}")
        print(f"Verdict     : {resp['response_verdict']}  |  Score: {resp['factuality_score']}")
        print(f"Text        : {text[:120]}...")
        print(f"\nExtracted {len(pred_claims)} claims from {len(annotations)} sentences:")

        tp = 0
        fp = 0
        for ann in annotations:
            sent = ann["sentence_text"]
            is_claim = ann["is_factual_claim"]
            in_gt = sent.strip().lower() in gt_texts_set
            
            if is_claim and in_gt:
                tp += 1
                match_tag = " <- MATCHED GT (TP)"
            elif is_claim and not in_gt:
                fp += 1
                match_tag = " <- FALSE POSITIVE"
            elif not is_claim and in_gt:
                match_tag = " <- MISSED GT (FN)"
            else:
                match_tag = ""

            tag   = "[CLAIM]" if is_claim else "[skip] "
            ctype = ann["claim_type"] or ""
            print(f"  {tag} ({ctype:20s}) {sent[:75]}{match_tag}")

        fn = len(gt_claims) - tp
        print(f"\n  Ground-truth claims: {len(gt_claims)} | Predicted: {len(pred_claims)} | TP: {tp} | FP: {fp} | FN: {fn}")

        total_gt        += len(gt_claims)
        total_predicted += len(pred_claims)
        total_correct   += tp

    print(f"\n{'=' * 70}")
    print(f"OVERALL ACROSS {args.n} RESPONSES")
    print(f"  Ground-truth claims : {total_gt}")
    print(f"  Predicted claims    : {total_predicted}")
    print(f"  True Positives (TP) : {total_correct}")
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall    = total_correct / total_gt        if total_gt > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    print(f"  Precision           : {precision:.4f}")
    print(f"  Recall              : {recall:.4f}")
    print(f"  F1 (exact match)    : {f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
