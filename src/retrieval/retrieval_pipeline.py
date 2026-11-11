"""
src/retrieval/retrieval_pipeline.py
-------------------------------------
High-level retrieval pipeline that combines:
  1. Dense retrieval (VectorStore)
  2. Cross-encoder reranking (EvidenceReranker)

into a single retrieve_evidence(claim) call that returns the best
passage to feed into the NLI module.
"""

import os
from typing import List, Dict, Any, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from src.retrieval.embedder import PassageEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import EvidenceReranker


class RetrievalPipeline:
    """
    Full retrieval pipeline:  claim -> [dense retrieval] -> [reranking] -> evidence

    Usage:
        pipeline = RetrievalPipeline()
        pipeline.build_index()   # once; persisted to disk

        evidence = pipeline.retrieve_evidence("Python was created by Guido van Rossum.")
        print(evidence["passage_text"])
        print(evidence["score"])         # dense retrieval cosine similarity
        print(evidence["rerank_score"])  # cross-encoder relevance
    """

    def __init__(
        self,
        embedder: PassageEmbedder = None,
        dense_top_k: int = 10,
        rerank_top_k: int = 3,
        use_reranker: bool = True,
    ):
        self.embedder = embedder or PassageEmbedder()
        self.store = VectorStore(embedder=self.embedder)
        self.reranker = EvidenceReranker() if use_reranker else None
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k

    def build_index(self, force: bool = False) -> None:
        """Build the ChromaDB index from the passage pool. Skips if already built."""
        self.store.build_index(force=force)

    def retrieve_evidence(
        self,
        claim: str,
        top_k: int = None,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve the best evidence passage for a claim.

        Args:
            claim:                 The atomic factual claim to verify.
            top_k:                 Override number of candidates returned from dense search.
            return_all_candidates: If True, return a list of all candidates instead of best.

        Returns:
            If return_all_candidates=False (default):
                Dict with passage_id, title, topic, passage_text, score, rerank_score.
            If return_all_candidates=True:
                List of candidate dicts sorted by rerank_score (or score if no reranker).
        """
        k = top_k or self.dense_top_k
        candidates = self.store.retrieve(claim, top_k=k)

        if not candidates:
            return {} if not return_all_candidates else []

        if self.reranker is not None:
            candidates = self.reranker.rerank(claim, candidates, top_k=self.rerank_top_k)

        if return_all_candidates:
            return candidates

        return candidates[0]

    def retrieve_evidence_batch(
        self,
        claims: List[str],
        return_all_candidates: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run retrieve_evidence() for a list of claims.

        Returns:
            List of results, one per claim.
        """
        return [
            self.retrieve_evidence(c, return_all_candidates=return_all_candidates)
            for c in claims
        ]

    def evaluate_retrieval(
        self,
        eval_data: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation on layer2_retrieval/retrieval_test.csv.
        Reports Recall@1, Recall@5, Recall@10, MRR, and nDCG@5, nDCG@10.
        """
        import math
        from collections import defaultdict

        query_to_gold = defaultdict(set)
        query_to_claim = {}
        for row in eval_data:
            cid = row["claim_id"]
            query_to_claim[cid] = row["claim_text"]
            if str(row.get("relevance_label", "0")) == "1":
                query_to_gold[cid].add(row["candidate_passage_text"].strip().lower()[:100])

        total_queries = len(query_to_claim)
        recalls_at_1 = 0
        recalls_at_5 = 0
        recalls_at_10 = 0
        reciprocal_ranks = []
        ndcg_5_list = []
        ndcg_10_list = []

        for cid, claim in query_to_claim.items():
            if not query_to_gold[cid]:
                total_queries -= 1
                continue

            candidates = self.store.retrieve(claim, top_k=10)
            if self.reranker:
                candidates = self.reranker.rerank(claim, candidates, top_k=10)

            found_rank = None
            dcg_5 = 0.0
            dcg_10 = 0.0

            for rank, cand in enumerate(candidates, start=1):
                cand_prefix = cand["passage_text"].strip().lower()[:100]
                is_rel = 1 if cand_prefix in query_to_gold[cid] else 0

                if is_rel and found_rank is None:
                    found_rank = rank

                if rank <= 5 and is_rel:
                    dcg_5 += 1.0 / math.log2(rank + 1)
                if rank <= 10 and is_rel:
                    dcg_10 += 1.0 / math.log2(rank + 1)

            if found_rank is not None:
                reciprocal_ranks.append(1.0 / found_rank)
                if found_rank <= 1:
                    recalls_at_1 += 1
                if found_rank <= 5:
                    recalls_at_5 += 1
                if found_rank <= 10:
                    recalls_at_10 += 1
            else:
                reciprocal_ranks.append(0.0)

            # IDCG for 1 relevant document is 1.0 / log2(2) = 1.0
            idcg = 1.0
            ndcg_5_list.append(dcg_5 / idcg)
            ndcg_10_list.append(dcg_10 / idcg)

        n = total_queries if total_queries > 0 else 1
        return {
            "recall@1": round(recalls_at_1 / n, 4),
            "recall@5": round(recalls_at_5 / n, 4),
            "recall@10": round(recalls_at_10 / n, 4),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "ndcg@5": round(sum(ndcg_5_list) / len(ndcg_5_list), 4) if ndcg_5_list else 0.0,
            "ndcg@10": round(sum(ndcg_10_list) / len(ndcg_10_list), 4) if ndcg_10_list else 0.0,
            "evaluated_queries": total_queries,
        }
