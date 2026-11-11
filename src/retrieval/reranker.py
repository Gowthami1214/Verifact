"""
src/retrieval/reranker.py
--------------------------
Cross-encoder reranker for TruthSync evidence selection.

After dense retrieval returns top-k candidates, the cross-encoder
jointly encodes (claim, evidence) pairs and scores them. The
highest-scoring passage becomes the gold evidence passed to the NLI stage.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Fast CPU inference (6-layer MiniLM)
  - Outputs a single relevance logit (higher = more relevant)
"""

import os
from typing import List, Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers.cross_encoder import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EvidenceReranker:
    """
    Cross-encoder reranker for (claim, evidence) pairs.

    Usage:
        reranker = EvidenceReranker()
        candidates = [
            {"passage_text": "...", "passage_id": "P0001", ...},
            ...
        ]
        ranked = reranker.rerank("Neil Armstrong landed on the Moon in 1969.", candidates)
        best_evidence = ranked[0]
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, cache_dir: str = None):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, cache_dir=cache_dir)
        print(f"[Reranker] Ready.")

    def rerank(
        self,
        claim: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Score and rerank a list of candidate passages for a given claim.

        Args:
            claim:      The factual claim text to match against.
            candidates: List of candidate dicts (must contain 'passage_text').
            top_k:      If set, return only top_k passages. Otherwise return all.

        Returns:
            Candidates list sorted by descending cross-encoder relevance score.
            Each dict gains a 'rerank_score' field (float).
        """
        if not candidates:
            return []

        pairs = [(claim, c["passage_text"]) for c in candidates]
        scores = self.model.predict(pairs)  # np.ndarray shape (N,)

        # Attach scores and sort descending
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = round(float(score), 4)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def select_best(
        self, claim: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Return the single best-ranked evidence passage for the given claim.

        Returns an empty dict if candidates list is empty.
        """
        ranked = self.rerank(claim, candidates, top_k=1)
        return ranked[0] if ranked else {}
