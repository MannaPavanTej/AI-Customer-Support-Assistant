"""
FAISS Vector Store module for indexing and searching knowledge base chunks.
Uses faiss.IndexFlatIP for exact cosine similarity search on normalized vectors.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import faiss
import numpy as np
import config
from .chunker import Chunk
from .embeddings import compute_embeddings


class FAISSVectorStore:
    """
    Manages FAISS index creation, persistence, loading, and similarity retrieval.
    """

    def __init__(self, index_dir: Path = config.VECTOR_STORE_DIR):
        self.index_dir = index_dir
        self.index_path = index_dir / "index.faiss"
        self.metadata_path = index_dir / "metadata.pkl"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []

    def build_from_chunks(self, chunks: List[Chunk]) -> int:
        """
        Computes embeddings for chunks, constructs FAISS index, and saves to disk.

        Args:
            chunks: List of Chunk objects

        Returns:
            Number of indexed chunks
        """
        if not chunks:
            return 0

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = compute_embeddings(texts)

        # Dimension of embedding (384 for all-MiniLM-L6-v2)
        dim = embeddings.shape[1]
        
        # Inner Product index on normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Persist index and chunk metadata
        self.save()
        return len(chunks)

    def save(self) -> None:
        """Saves FAISS index and chunk metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> bool:
        """
        Loads FAISS index and metadata from disk if they exist.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as f:
                    self.chunks = pickle.load(f)
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                return False
        return False

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Performs cosine similarity search against the index.

        Args:
            query_embedding: L2-normalized 1D or 2D vector
            top_k: Number of nearest neighbors to retrieve

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
