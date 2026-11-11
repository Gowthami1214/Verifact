"""
scripts/run_pipeline.py
-------------------------
Interactive TruthSync demo: type any AI-generated text and get a
factuality verification report instantly.

Also supports batch evaluation on the Layer 5 held-out benchmark and
saves a full evaluation JSON.

Usage:
    # Interactive mode
    python scripts/run_pipeline.py

    # Benchmark on held-out responses (first N)
    python scripts/run_pipeline.py --eval --n 50

    # Use specific examples
    python scripts/run_pipeline.py --text "Neil Armstrong landed on the Moon in 1972. He was the commander of Apollo 11."
"""

import os, sys, csv, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"]   = "1"

from src.pipeline import TruthSyncPipeline

RESPONSES_PATH = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_responses.csv")
CLAIMS_PATH    = os.path.join(ROOT, "dataset", "layer5_heldout", "heldout_claims.csv")
EVAL_OUT_PATH  = os.path.join(ROOT, "models", "eval_results.json")


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_interactive(pipeline: TruthSyncPipeline):
    print("\nTruthSync Interactive Mode")
    print("Enter an AI-generated text to verify (or 'quit' to exit).\n")
    while True:
        try:
            text = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        report = pipeline.verify(text)
        print("\n" + pipeline.format_report(report) + "\n")


def run_benchmark(pipeline: TruthSyncPipeline, n: int):
    """Run on Layer 5 held-out set and compute claim and response accuracy."""
    responses  = load_csv(RESPONSES_PATH)[:n]
    claims_all = load_csv(CLAIMS_PATH)

    # Index ground truth by response_id
    gt_claims_by_resp = {}
    for c in claims_all:
        gt_claims_by_resp.setdefault(c["response_id"], []).append(c)

    print(f"\nBenchmarking on {len(responses)} held-out responses...\n")

    total_claims    = 0
    correct_claims  = 0
    correct_verdict = 0
    results_log     = []

    for resp in responses:
        resp_id   = resp["response_id"]
        text      = resp["generated_text"]
        gt_verdict = resp["response_verdict"]
        gt_score   = float(resp["factuality_score"])

        # Build gold label map keyed by claim text
        gt_map = {c["claim_text"].strip(): c["label"]
                  for c in gt_claims_by_resp.get(resp_id, [])}

        report = pipeline.verify(text)

        # Match predicted claims to gold by text
        for pred_claim in report["claims"]:
            gold_label = gt_map.get(pred_claim["claim_text"].strip())
            if gold_label:
                total_claims += 1
                if pred_claim["label"] == gold_label:
                    correct_claims += 1

        pred_verdict = report["response_verdict"]
        if pred_verdict == gt_verdict:
            correct_verdict += 1

        results_log.append({
            "response_id":     resp_id,
            "predicted_verdict": pred_verdict,
            "gold_verdict":      gt_verdict,
            "predicted_score":   report["factuality_score"],
            "gold_score":        gt_score,
            "claims":            report["claims"],
        })

        print(f"  {resp_id} | pred={report['factuality_score']:.2f} '{pred_verdict[:22]}'  "
              f"| gold={gt_score:.2f} '{gt_verdict}'")

    # Summary
    claim_acc   = correct_claims  / total_claims    if total_claims    > 0 else 0.0
    verdict_acc = correct_verdict / len(responses)  if responses        else 0.0

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS ({n} responses)")
    print(f"{'=' * 60}")
    print(f"  Claim-level accuracy   : {correct_claims}/{total_claims} = {claim_acc:.2%}")
    print(f"  Response-level verdict : {correct_verdict}/{len(responses)} = {verdict_acc:.2%}")

    summary = {
        "n_responses":       len(responses),
        "claim_accuracy":    round(claim_acc, 4),
        "verdict_accuracy":  round(verdict_acc, 4),
        "correct_claims":    correct_claims,
        "total_claims":      total_claims,
        "correct_verdicts":  correct_verdict,
        "results":           results_log,
    }
    os.makedirs(os.path.dirname(EVAL_OUT_PATH), exist_ok=True)
    with open(EVAL_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"\nFull results saved to: {EVAL_OUT_PATH}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="TruthSync Pipeline Runner")
    parser.add_argument("--text",  type=str, default=None,
                        help="Single text to verify")
    parser.add_argument("--demo", action="store_true",
                        help="Run built-in demo examples and exit")
    parser.add_argument("--eval",  action="store_true",
                        help="Run benchmark evaluation on held-out set")
    parser.add_argument("--n",     type=int, default=30,
                        help="Number of held-out responses to evaluate (default: 30)")
    parser.add_argument("--nli",   default=None, choices=["pretrained", "finetuned"],
                        help="NLI mode override")
    args = parser.parse_args()

    pipeline = TruthSyncPipeline(nli_mode=args.nli)

    if args.text:
        report = pipeline.verify(args.text)
        print("\n" + pipeline.format_report(report))
    elif args.eval:
        run_benchmark(pipeline, n=args.n)
    elif args.demo or not sys.stdin.isatty():
        demos = [
            "Tim Berners-Lee invented the World Wide Web in 1989 at CERN. "
            "He also invented the Python programming language in 1991.",

            "The Berlin Wall was built in 1961 and fell on November 9, 1989. "
            "It divided the city of Paris during the Cold War.",

            "Marie Curie was the first woman to win a Nobel Prize. "
            "She received two Nobel Prizes, one in Physics and one in Chemistry.",
        ]
        print("\n=== TruthSync Built-in Demo Examples ===")
        for demo_text in demos:
            report = pipeline.verify(demo_text)
            print("\n" + pipeline.format_report(report))
    else:
        demos = [
            "Tim Berners-Lee invented the World Wide Web in 1989 at CERN. "
            "He also invented the Python programming language in 1991.",

            "The Berlin Wall was built in 1961 and fell on November 9, 1989. "
            "It divided the city of Paris during the Cold War.",

            "Marie Curie was the first woman to win a Nobel Prize. "
            "She received two Nobel Prizes, one in Physics and one in Chemistry.",
        ]
        print("\n=== TruthSync Built-in Demo Examples ===")
        for demo_text in demos:
            report = pipeline.verify(demo_text)
            print("\n" + pipeline.format_report(report))

        run_interactive(pipeline)


if __name__ == "__main__":
    main()
