"""
src/pipeline.py
-----------------
TruthSync End-to-End Factuality Verification Pipeline.

Orchestrates all three modules in sequence:
  1. ClaimExtractor  (Module 2) — split AI response into atomic claims
  2. RetrievalPipeline (Module 1) — retrieve best evidence per claim
  3. NLIClassifier   (Module 3) — verify claim against evidence
  4. Aggregator      — compute response-level factuality score and verdict

Usage:
    from src.pipeline import TruthSyncPipeline

    pipeline = TruthSyncPipeline()

    report = pipeline.verify(
        "Python was invented by Guido van Rossum in 1991. "
        "It was first developed at MIT. "
        "Python 3 was released in 2008."
    )

    print(report["response_verdict"])    # "Mixed factuality"
    print(report["factuality_score"])    # 0.67
    for claim in report["claims"]:
        print(claim["claim_text"], "->", claim["label"], f"({claim['confidence']:.2f})")
"""

import os
from typing import List, Dict, Any, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from src.claim_extraction.claim_extractor import ClaimExtractor
from src.retrieval.retrieval_pipeline      import RetrievalPipeline
from src.nli.nli_classifier                import NLIClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FINETUNED_NLI = os.path.join(_ROOT, "models", "nli", "deberta-v3-small-nli")


class TruthSyncPipeline:
    """
    Full TruthSync factuality verification pipeline.

    Args:
        extraction_mode:  "rule" | "nli" | "trained"  (default: "rule")
        nli_mode:         "finetuned" | "pretrained"   (default: auto-detects)
        dense_top_k:      candidates from vector search (default: 10)
        rerank_top_k:     candidates after cross-encoder reranking (default: 3)
        use_reranker:     enable cross-encoder reranking (default: True)
    """

    def __init__(
        self,
        extraction_mode: str = "rule",
        nli_mode: str = None,
        dense_top_k: int = 10,
        rerank_top_k: int = 3,
        use_reranker: bool = True,
    ):
        # ── Auto-detect NLI mode ──────────────────────────────────────────────
        if nli_mode is None:
            nli_mode = "finetuned" if os.path.isdir(_FINETUNED_NLI) else "pretrained"

        print("[TruthSync] Initializing pipeline...")
        print(f"  Claim extraction : {extraction_mode}")
        print(f"  NLI mode         : {nli_mode}")
        print(f"  Reranker         : {'ON' if use_reranker else 'OFF'}")

        self.extractor = ClaimExtractor(mode=extraction_mode)
        self.retrieval = RetrievalPipeline(
            dense_top_k=dense_top_k,
            rerank_top_k=rerank_top_k,
            use_reranker=use_reranker,
        )
        self.nli = NLIClassifier(mode=nli_mode)
        self.retrieval.build_index()   # no-op if already built
        print("[TruthSync] Pipeline ready.\n")

    # ── Core API ──────────────────────────────────────────────────────────────

    def verify(
        self,
        text: str,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify the factual accuracy of an AI-generated text response.

        Args:
            text:                  The full AI-generated response paragraph.
            return_all_candidates: Include all retrieved candidates per claim.

        Returns:
            {
              "input_text":        str,
              "factuality_score":  float,       # 0.0–1.0, supports/total
              "response_verdict":  str,          # human-readable verdict
              "supports_count":    int,
              "refutes_count":     int,
              "nei_count":         int,
              "total_claims":      int,
              "claims": [
                {
                  "sentence_id":    int,
                  "claim_text":     str,
                  "claim_type":     str,
                  "evidence_text":  str,
                  "evidence_score": float,       # retrieval score
                  "label":          str,          # SUPPORTS | REFUTES | NOT_ENOUGH_INFO
                  "confidence":     float,
                  "scores":         dict,
                  "candidates":     list | None   # if return_all_candidates=True
                }
              ]
            }
        """
        # Step 1: Extract claims
        clean_text = text.strip().strip('"\'“”‘’')
        annotations = self.extractor.extract(clean_text)
        claim_annotations = [a for a in annotations if a["is_factual_claim"]]

        if not claim_annotations:
            # Fallback: if user provided valid sentences, verify them as claims
            non_empty = [a for a in annotations if len(a.get("sentence_text", "").split()) >= 3]
            if non_empty:
                for a in non_empty:
                    a["is_factual_claim"] = True
                    if not a.get("claim_type"):
                        a["claim_type"] = "general"
                claim_annotations = non_empty
            else:
                return self._empty_report(text, "No factual claims detected.")

        claim_texts = [a["sentence_text"] for a in claim_annotations]

        # Step 2 & 3: Retrieve + NLI for each claim using multi-evidence aggregation
        claim_results = []
        for ann in claim_annotations:
            claim_text = ann["sentence_text"]

            # Retrieve top candidates
            candidates = self.retrieval.retrieve_evidence(
                claim_text, return_all_candidates=True
            )

            # Transparent multi-evidence aggregation
            nli_decision = self._aggregate_candidate_evidences(candidates, claim_text)
            best_ev = nli_decision["best_ev"]
            ev_text = best_ev.get("passage_text", "")
            ev_score = best_ev.get("rerank_score", best_ev.get("score", 0.0))

            claim_results.append({
                "sentence_id":    ann["sentence_id"],
                "claim_text":     claim_text,
                "claim_type":     ann.get("claim_type", ""),
                "evidence_text":  ev_text,
                "evidence_score": round(float(ev_score), 4),
                "label":          nli_decision["label"],
                "confidence":     nli_decision["confidence"],
                "scores":         nli_decision["scores"],
                "candidates":     candidates if return_all_candidates else None,
            })

        # Step 4: Aggregate factuality
        agg = self.nli.score_factuality([{"label": r["label"]} for r in claim_results])

        return {
            "input_text":       text,
            "factuality_score": agg["factuality_score"],
            "response_verdict": agg["response_verdict"],
            "supports_count":   agg["supports_count"],
            "refutes_count":    agg["refutes_count"],
            "nei_count":        agg["nei_count"],
            "total_claims":     agg["total_claims"],
            "claims":           claim_results,
        }

    def _aggregate_candidate_evidences(self, candidates: List[Dict], claim_text: str) -> Dict:
        """
        Evaluate NLI across top-k retrieved candidates and aggregate transparently:
          1. Contradiction Precedence: If ANY relevant candidate refutes the claim (conf >= 0.45),
             the claim is marked REFUTES.
          2. Verified Support: Else if ANY candidate supports the claim (conf >= 0.45),
             the claim is marked SUPPORTS.
          3. Otherwise: Highest confidence candidate.
        """
        if not candidates:
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "scores": {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT_ENOUGH_INFO": 1.0},
                "best_ev": {
                    "passage_text": "No relevant factual evidence found in the knowledge base.",
                    "passage_id": "NONE",
                    "rerank_score": -10.0,
                },
            }

        # Relevance Gating: In MS-MARCO Cross-Encoder, scores < -2.5 indicate zero topical relevance
        RELEVANCE_THRESHOLD = -2.5
        top_cand = candidates[0]
        top_score = top_cand.get("rerank_score", top_cand.get("score", 0.0))

        if top_score < RELEVANCE_THRESHOLD:
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 0.95,
                "scores": {"SUPPORTS": 0.02, "REFUTES": 0.03, "NOT_ENOUGH_INFO": 0.95},
                "best_ev": {
                    "passage_text": "No relevant factual evidence found in the knowledge base to corroborate or refute this claim.",
                    "passage_id": "NONE",
                    "rerank_score": top_score,
                },
            }

        # Filter only semantically relevant candidates
        evaluated = []
        for cand in candidates[:3]:
            score = cand.get("rerank_score", cand.get("score", 0.0))
            if score < RELEVANCE_THRESHOLD:
                continue
            ev_text = cand.get("passage_text", "")
            if not ev_text:
                continue
            pred = self.nli.predict(ev_text, claim_text)
            evaluated.append({"candidate": cand, "pred": pred})

        if not evaluated:
            return {
                "label": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "scores": {"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT_ENOUGH_INFO": 1.0},
                "best_ev": candidates[0] if candidates else {},
            }

        # Check for supported and refuted candidates
        sup_candidates = [e for e in evaluated if e["pred"]["label"] == "SUPPORTS" and e["pred"]["confidence"] >= 0.45]
        ref_candidates = [e for e in evaluated if e["pred"]["label"] == "REFUTES" and e["pred"]["confidence"] >= 0.45]

        # Prioritize confident support when available and stronger than refutation
        if sup_candidates and not ref_candidates:
            best_sup = max(sup_candidates, key=lambda x: x["pred"]["confidence"])
            return {
                "label": "SUPPORTS",
                "confidence": best_sup["pred"]["confidence"],
                "scores": best_sup["pred"]["scores"],
                "best_ev": best_sup["candidate"],
            }

        # If both exist, prioritize decisive refutation only if confidence >= 0.60
        if ref_candidates and max(c["pred"]["confidence"] for c in ref_candidates) >= 0.60:
            best_ref = max(ref_candidates, key=lambda x: x["pred"]["confidence"])
            return {
                "label": "REFUTES",
                "confidence": best_ref["pred"]["confidence"],
                "scores": best_ref["pred"]["scores"],
                "best_ev": best_ref["candidate"],
            }
        elif sup_candidates:
            best_sup = max(sup_candidates, key=lambda x: x["pred"]["confidence"])
            return {
                "label": "SUPPORTS",
                "confidence": best_sup["pred"]["confidence"],
                "scores": best_sup["pred"]["scores"],
                "best_ev": best_sup["candidate"],
            }
        elif ref_candidates:
            best_ref = max(ref_candidates, key=lambda x: x["pred"]["confidence"])
            return {
                "label": "REFUTES",
                "confidence": best_ref["pred"]["confidence"],
                "scores": best_ref["pred"]["scores"],
                "best_ev": best_ref["candidate"],
            }

        best_overall = max(evaluated, key=lambda x: x["pred"]["confidence"])
        return {
            "label": best_overall["pred"]["label"],
            "confidence": best_overall["pred"]["confidence"],
            "scores": best_overall["pred"]["scores"],
            "best_ev": best_overall["candidate"],
        }

    def verify_batch(
        self, texts: List[str], return_all_candidates: bool = False
    ) -> List[Dict[str, Any]]:
        """Verify a list of AI-generated responses. Returns list of verify() outputs."""
        return [self.verify(t, return_all_candidates=return_all_candidates) for t in texts]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_report(self, text: str, note: str = "") -> Dict[str, Any]:
        return {
            "input_text":       text,
            "factuality_score": 0.0,
            "response_verdict": "No verifiable claims",
            "supports_count":   0,
            "refutes_count":    0,
            "nei_count":        0,
            "total_claims":     0,
            "claims":           [],
            "note":             note,
        }

    def format_report(self, report: Dict[str, Any]) -> str:
        """Pretty-print a verify() report as an ASCII string."""
        lines = [
            "=" * 70,
            "TruthSync Factuality Verification Report",
            "=" * 70,
            f"Response Verdict : {report['response_verdict']}",
            f"Factuality Score : {report['factuality_score']:.2%}",
            f"Claims Breakdown : {report['supports_count']} SUPPORTS  "
            f"{report['refutes_count']} REFUTES  "
            f"{report['nei_count']} NOT_ENOUGH_INFO  "
            f"(total: {report['total_claims']})",
            "",
            "Claim-by-Claim Results:",
            "-" * 70,
        ]
        for i, c in enumerate(report["claims"], 1):
            label_icon = {"SUPPORTS": "[+]", "REFUTES": "[-]", "NOT_ENOUGH_INFO": "[?]"}.get(
                c["label"], "[ ]"
            )
            lines.append(
                f"{i:2d}. {label_icon} ({c['confidence']:.2f}) [{c['claim_type']:15s}] "
                f"{c['claim_text'][:65]}"
            )
            lines.append(
                f"     Evidence: {c['evidence_text'][:65]}  [score={c['evidence_score']:.3f}]"
            )
        lines.append("=" * 70)
        return "\n".join(lines)
