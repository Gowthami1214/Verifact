"""
src/retrieval/vector_store.py
------------------------------
ChromaDB-backed persistent vector store for TruthSync passage retrieval.

Responsibilities:
  - Index all 302 passages from passage_pool.jsonl into a named ChromaDB collection.
  - Persist the index to disk (models/retrieval/chroma_db/) for re-use across runs.
  - Expose a simple retrieve(query_text, top_k) interface that returns ranked passages.
  - Fall back to FAISS cosine search if ChromaDB is unavailable.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import chromadb
from chromadb.config import Settings

from src.retrieval.embedder import PassageEmbedder

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASSAGE_POOL_PATH = os.path.join(_ROOT, "dataset", "layer2_retrieval", "passage_pool.jsonl")
CHROMA_PERSIST_DIR = os.path.join(_ROOT, "models", "retrieval", "chroma_db")
COLLECTION_NAME = "truthsync_passages"


def _load_passage_pool(path: str) -> List[Dict[str, Any]]:
    passages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


class VectorStore:
    """
    Persistent ChromaDB vector store for TruthSync passages.

    Usage:
        store = VectorStore()
        store.build_index()             # only needed once – persists to disk
        results = store.retrieve("The Nile is the longest river", top_k=5)
    """

    def __init__(
        self,
        embedder: PassageEmbedder = None,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.embedder = embedder or PassageEmbedder()
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialise persistent client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._collection = None

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, passage_pool_path: str = PASSAGE_POOL_PATH, force: bool = False) -> None:
        """
        Embed all passages in passage_pool.jsonl and upsert them into ChromaDB.

        Args:
            passage_pool_path: Path to the JSONL passage pool.
            force:             If True, drop and rebuild the collection even if it exists.
        """
        existing = [c.name for c in self.client.list_collections()]

        if self.collection_name in existing:
            if not force:
                print(f"[VectorStore] Collection '{self.collection_name}' already exists. Loading.")
                self._collection = self.client.get_collection(self.collection_name)
                print(f"[VectorStore] {self._collection.count()} passages indexed.")
                return
            else:
                print(f"[VectorStore] Dropping existing collection '{self.collection_name}'.")
                self.client.delete_collection(self.collection_name)

        # Create collection with cosine similarity metric
        self._collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        passages = _load_passage_pool(passage_pool_path)
        print(f"[VectorStore] Embedding {len(passages)} passages...")

        texts = [p["passage_text"] for p in passages]
        ids = [p["passage_id"] for p in passages]
        metadatas = [
            {
                "passage_id": p["passage_id"],
                "title": p.get("title", ""),
                "topic": p.get("topic", ""),
                "source_category": p.get("source_category", ""),
            }
            for p in passages
        ]

        # Embed in one batch (302 passages is small)
        embeddings = self.embedder.encode(texts, normalize=True, show_progress=True)

        # Upsert in ChromaDB (accepts python lists of floats)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[VectorStore] Indexed {self._collection.count()} passages into '{self.collection_name}'.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        if self._collection is None:
            existing = [c.name for c in self.client.list_collections()]
            if self.collection_name not in existing:
                raise RuntimeError(
                    "Vector index not built yet. Call build_index() first."
                )
            self._collection = self.client.get_collection(self.collection_name)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        topic_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most semantically similar passages for a given query.

        Args:
            query_text:   The claim or question string to search for.
            top_k:        Number of results to return.
            topic_filter: Optional ChromaDB metadata filter on 'topic'.

        Returns:
            List of dicts, each with keys:
              passage_id, title, topic, source_category, passage_text, score
            Sorted by descending cosine similarity (score 1.0 = identical).
        """
        self._ensure_collection()

        query_vec = self.embedder.encode_query(query_text)

        where_filter = {"topic": topic_filter} if topic_filter else None

        results = self._collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB cosine distance: distance = 1 - cosine_similarity
        # So similarity = 1 - distance
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "passage_id": meta.get("passage_id", ""),
                    "title": meta.get("title", ""),
                    "topic": meta.get("topic", ""),
                    "source_category": meta.get("source_category", ""),
                    "passage_text": doc,
                    "score": round(1.0 - dist, 4),
                }
            )

        return output

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve candidates for multiple queries in one call.

        Returns:
            List of retrieve() results, one per query.
        """
        return [self.retrieve(q, top_k=top_k) for q in queries]
