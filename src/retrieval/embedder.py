"""
src/retrieval/embedder.py
-------------------------
Sentence embedding module for TruthSync.

Wraps the sentence-transformers SentenceTransformer to produce
unit-normed dense embeddings used for cosine similarity search.

Model: all-MiniLM-L6-v2 (default, 384-dim, fast CPU inference)
       Can be swapped to BAAI/bge-base-en-v1.5 for higher accuracy.
"""

import os
import numpy as np
from typing import List, Union

# Suppress TF/Keras noise – we only use PyTorch backend
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class PassageEmbedder:
    """
    Thin wrapper around SentenceTransformer for encoding claims and passages.

    Usage:
        embedder = PassageEmbedder()
        vecs = embedder.encode(["claim 1", "passage text"])  # np.ndarray (N, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = None):
        print(f"[Embedder] Loading model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Embedding dimension: {self.dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into dense vectors.

        Args:
            texts:         A string or list of strings to encode.
            batch_size:    Number of texts per encoding batch.
            normalize:     If True, L2-normalizes outputs (needed for cosine similarity).
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), self.dim), dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single claim/query string. Returns shape (1, dim)."""
        return self.encode([query], normalize=True)
