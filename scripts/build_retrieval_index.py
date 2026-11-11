"""
scripts/build_retrieval_index.py
----------------------------------
One-time script: embeds the 302-passage pool with all-MiniLM-L6-v2
and persists the ChromaDB index to models/retrieval/chroma_db/.

Run from the TruthSync root:
    python scripts/build_retrieval_index.py

After running, the index is reused automatically by retrieval_pipeline.py.
"""

import os, sys

# Ensure project root is on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.retrieval.retrieval_pipeline import RetrievalPipeline

if __name__ == "__main__":
    print("=" * 60)
    print("TruthSync - Building Retrieval Index")
    print("=" * 60)

    pipeline = RetrievalPipeline(
        dense_top_k=10,
        rerank_top_k=3,
        use_reranker=False,  # Don't load reranker just for indexing
    )

    print("\n[Step 1] Building ChromaDB vector index from passage_pool.jsonl...")
    pipeline.build_index(force=False)  # Set force=True to rebuild from scratch

    print("\n[Step 2] Quick smoke test -- 3 sample queries:")
    test_queries = [
        "Who invented Python programming language?",
        "When was the Berlin Wall constructed?",
        "What is the deepest lake in the world?",
    ]

    for q in test_queries:
        results = pipeline.store.retrieve(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(f"    [{r['score']:.4f}] [{r['topic']}] {r['passage_text'][:80]}...")

    print("\nIndex build complete. ChromaDB persisted to models/retrieval/chroma_db/")
