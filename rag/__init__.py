"""
RAG (Retrieval-Augmented Generation) package.
Provides document loading, chunking, embedding generation, FAISS indexing, and retrieval.
"""

from .document_loader import load_documents, Document
from .chunker import chunk_documents, Chunk
from .embeddings import get_embedding_model, compute_embeddings
from .vector_store import FAISSVectorStore
from .retriever import KnowledgeRetriever

__all__ = [
    "load_documents",
    "Document",
    "chunk_documents",
    "Chunk",
    "get_embedding_model",
    "compute_embeddings",
    "FAISSVectorStore",
    "KnowledgeRetriever"
]
