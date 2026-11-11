"""
scripts/evaluate_retrieval.py
-------------------------------
Evaluates the retrieval module on layer2_retrieval/retrieval_test.csv.

Metrics reported:
  - Hit Rate@k: proportion of queries where gold evidence is in top-k results
  - MRR@k: mean reciprocal rank

Usage:
    python scripts/evaluate_retrieval.py [--top-k 5] [--no-reranker]

Results are printed and also saved to models/retrieval/retrieval_eval_results.json
"""

import os, sys, json, csv, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.retrieval.retrieval_pipeline import RetrievalPipeline

RETRIEVAL_TEST_PATH = os.path.join(ROOT, "dataset", "layer2_retrieval", "retrieval_test.csv")
RESULTS_PATH = os.path.join(ROOT, "models", "retrieval", "retrieval_eval_results.json")


def load_retrieval_test(path: str):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5, help="Hit@k threshold (default: 5)")
    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    print("=" * 60)
    print("TruthSync - Retrieval Evaluation")
    print(f"  top_k={args.top_k}  reranker={'OFF' if args.no_reranker else 'ON'}")
    print("=" * 60)

    use_reranker = not args.no_reranker
    pipeline = RetrievalPipeline(
        dense_top_k=args.top_k * 2,  # Fetch 2x for reranker to work with
        rerank_top_k=args.top_k,
        use_reranker=use_reranker,
    )
    pipeline.build_index()  # No-op if already built

    print(f"\nLoading retrieval test set from: {RETRIEVAL_TEST_PATH}")
    eval_data = load_retrieval_test(RETRIEVAL_TEST_PATH)
    print(f"  Loaded {len(eval_data)} candidate rows.")

    print(f"\nRunning evaluation (Hit@{args.top_k} and MRR@{args.top_k})...")
    metrics = pipeline.evaluate_retrieval(eval_data, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out_dict = {
        "model": "all-MiniLM-L6-v2",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else "none",
        **metrics,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    
    deliverables_path = os.path.join(ROOT, "results", "retrieval_metrics.json")
    os.makedirs(os.path.dirname(deliverables_path), exist_ok=True)
    with open(deliverables_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Deliverable saved to: {deliverables_path}")


if __name__ == "__main__":
    main()
