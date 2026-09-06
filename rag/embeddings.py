"""
Embedding Generation Service for RAG vector search.
Uses SentenceTransformers to encode text into dense, normalized vector representations.
"""

from typing import List, Union
import hashlib
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import config

# Singleton instance to prevent repeated model reloads
_EMBEDDING_MODEL = None
_MODEL_UNAVAILABLE = False
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional dense vector space, ultra-fast on CPU


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the cached singleton SentenceTransformer model instance.
    """
    global _EMBEDDING_MODEL, _MODEL_UNAVAILABLE
    if _EMBEDDING_MODEL is None:
        try:
            # Do not make normal app startup depend on an unannounced network
            # download.  Operators can cache this model during deployment.
            _EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            _MODEL_UNAVAILABLE = True
    return _EMBEDDING_MODEL


def _hash_embeddings(texts: List[str]) -> np.ndarray:
    """Create deterministic, local token-hash vectors as an offline fallback."""
    vectors = np.zeros((len(texts), config.EMBEDDING_DIMENSION), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in re.findall(r"[\w'-]+", text.lower()):
            bucket = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big") % config.EMBEDDING_DIMENSION
            vectors[row, bucket] += 1.0
        norm = np.linalg.norm(vectors[row])
        if norm:
            vectors[row] /= norm
    return vectors


def compute_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Computes L2-normalized dense vector embeddings for input text(s).

    Args:
        texts: A single text string or list of text strings

    Returns:
        np.ndarray: Matrix of shape (N, embedding_dim) with unit length vectors
    """
    if isinstance(texts, str):
        texts = [texts]

    model = get_embedding_model()
    if model is None:
        return _hash_embeddings(texts)

    # Generate embeddings and normalize to unit sphere (L2 norm = 1.0)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)
