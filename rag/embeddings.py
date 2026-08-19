"""
Embedding Generation Service for RAG vector search.
Uses SentenceTransformers to encode text into dense, normalized vector representations.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton instance to prevent repeated model reloads
_EMBEDDING_MODEL = None
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional dense vector space, ultra-fast on CPU


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the cached singleton SentenceTransformer model instance.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBEDDING_MODEL


def compute_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Computes L2-normalized dense vector embeddings for input text(s).

    Args:
        texts: A single text string or list of text strings

    Returns:
        np.ndarray: Matrix of shape (N, embedding_dim) with unit length vectors
    """
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]

    # Generate embeddings and normalize to unit sphere (L2 norm = 1.0)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)
