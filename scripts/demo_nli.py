"""
scripts/demo_nli.py
---------------------
Demonstrates the full TruthSync pipeline for 10 held-out claims:

  claim -> [retrieval] -> evidence -> [NLI] -> SUPPORTS/REFUTES/NEI + confidence

Uses:
  - RetrievalPipeline  (Module 1, pretrained all-MiniLM-L6-v2 + ChromaDB)
  - NLIClassifier      (Module 3, pretrained roberta-large-mnli, no fine-tuning needed)

The demo shows predicted vs. ground-truth labels and response-level factuality scores.

Usage:
    python scripts/demo_nli.py [--n 10] [--mode pretrained|finetuned]
"""

import os, sys, csv, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.nli.nli_classifier           import NLIClassifier

RESPONSES_PATH = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_responses.csv")
CLAIMS_PATH    = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_claims.csv")
EVIDENCE_PATH  = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_evidence.csv")


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10,
                        help="Number of claims to demo (default: 10)")
    parser.add_argument("--mode", default="pretrained",
                        choices=["pretrained", "finetuned"],
                        help="NLI mode (default: pretrained)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TruthSync - Full Pipeline Demo (NLI mode={args.mode})")
    print("=" * 70)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[1/2] Loading retrieval pipeline...")
    retrieval = RetrievalPipeline(dense_top_k=10, rerank_top_k=3, use_reranker=True)
    retrieval.build_index()

    print("\n[2/2] Loading NLI classifier...")
    nli = NLIClassifier(mode=args.mode)

    # ── Load held-out data ────────────────────────────────────────────────────
    claims    = load_csv(CLAIMS_PATH)[:args.n]
    responses = {r["response_id"]: r for r in load_csv(RESPONSES_PATH)}
    evidence  = load_csv(EVIDENCE_PATH)

    # Index gold evidence by claim_id
    gold_ev = {}
    for ev in evidence:
        if ev["relevance_label"] == "1":
            gold_ev[ev["claim_id"]] = ev["passage_text"]

    # ── Run per-claim pipeline ─────────────────────────────────────────────────
    correct        = 0
    per_resp_preds = {}

    print(f"\nRunning end-to-end on {args.n} claims...\n")

    for claim_row in claims:
        cid        = claim_row["claim_id"]
        resp_id    = claim_row["response_id"]
        claim_text = claim_row["claim_text"]
        gt_label   = claim_row["label"]
        ctype      = claim_row["claim_type"]
        difficulty = claim_row["difficulty"]

        # Step 1: Retrieve evidence
        retrieved  = retrieval.retrieve_evidence(claim_text)
        ret_text   = retrieved.get("passage_text", "")
        ret_score  = retrieved.get("rerank_score", retrieved.get("score", 0.0))

        # Step 2: NLI classification using retrieved evidence
        nli_result = nli.predict(ret_text, claim_text)
        pred_label = nli_result["label"]
        confidence = nli_result["confidence"]
        is_correct = pred_label == gt_label

        if is_correct:
            correct += 1

        # Accumulate per-response predictions
        per_resp_preds.setdefault(resp_id, []).append(nli_result)

        # Display
        status = "CORRECT" if is_correct else "WRONG  "
        print(f"[{status}] {cid}  ({ctype}, {difficulty})")
        print(f"  Claim      : {claim_text[:80]}")
        print(f"  Retrieved  : {ret_text[:80]}  [score={ret_score:.3f}]")
        print(f"  Predicted  : {pred_label:20s} (conf={confidence:.3f})")
        print(f"  Gold Label : {gt_label}")
        print()

    # ── Response-level Factuality Report ──────────────────────────────────────
    print("=" * 70)
    print("RESPONSE-LEVEL FACTUALITY REPORT")
    print("=" * 70)

    resp_correct = 0
    resp_total   = len(per_resp_preds)

    for resp_id, pred_list in sorted(per_resp_preds.items()):
        gt_resp      = responses.get(resp_id, {})
        gt_verdict   = gt_resp.get("response_verdict", "?")
        gt_score     = gt_resp.get("factuality_score", "?")
        agg          = nli.score_factuality(pred_list)
        pred_verdict = agg["response_verdict"]
        pred_score   = agg["factuality_score"]

        match = "(match)" if pred_verdict == gt_verdict else "(differ)"
        print(f"  {resp_id} | Predicted: {pred_score:.2f} '{pred_verdict:22s}'  "
              f"| Gold: {gt_score} '{gt_verdict}' {match}")

        if pred_verdict == gt_verdict:
            resp_correct += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    claim_acc = correct / args.n if args.n > 0 else 0.0
    resp_acc  = resp_correct / resp_total if resp_total > 0 else 0.0
    print(f"  Claim-level accuracy   : {correct}/{args.n} = {claim_acc:.2%}")
    print(f"  Response-level verdict : {resp_correct}/{resp_total} = {resp_acc:.2%}")
    print()


if __name__ == "__main__":
    main()
