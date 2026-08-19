"""
Knowledge Retriever Module for Grounded RAG.
Coordinates query embedding, FAISS similarity search, source attribution, and strict no-hallucination handling.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .document_loader import load_documents
from .chunker import chunk_documents
from .embeddings import compute_embeddings
from .vector_store import FAISSVectorStore

# Strict similarity threshold: Scores below this are treated as "Out-of-Scope / No Answer"
RELEVANCE_THRESHOLD = 0.35


class KnowledgeRetriever:
    """
    Orchestrates the complete RAG retrieval pipeline for TechCare knowledge grounding.
    """

    def __init__(self, vector_store: Optional[FAISSVectorStore] = None):
        self.vector_store = vector_store or FAISSVectorStore()
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Loads existing index from disk or builds a fresh index from knowledge base docs."""
        if not self.vector_store.load():
            print("Building fresh FAISS index from TechCare knowledge base...")
            docs = load_documents()
            chunks = chunk_documents(docs)
            self.vector_store.build_from_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Retrieves relevant company knowledge chunks for a customer query.

        Args:
            query: Customer question or message text
            top_k: Maximum number of relevant chunks to retrieve

        Returns:
            Dict containing:
            - found: bool (True if relevant chunks exceed threshold)
            - sources: List of source attribution dictionaries
            - context_text: Combined text for AI response grounding
            - fallback_message: Explicit disclaimer if no relevant document exists
        """
        if not query or not query.strip():
            return {
                "found": False,
                "sources": [],
                "context_text": "",
                "fallback_message": "No query provided."
            }

        # 1. Compute normalized query vector
        query_vector = compute_embeddings(query.strip())

        # 2. Search FAISS index
        matches = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Filter by similarity threshold (Zero-Hallucination Guardrail)
        valid_sources: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        for chunk, score in matches:
            if score >= RELEVANCE_THRESHOLD:
                valid_sources.append({
                    "document": chunk.document_title,
                    "section": chunk.section_title,
                    "source_file": chunk.source_file,
                    "category": chunk.category,
                    "similarity_score": round(score, 3),
                    "text": chunk.text
                })
                context_parts.append(chunk.text)

        if valid_sources:
            return {
                "found": True,
                "sources": valid_sources,
                "context_text": "\n\n---\n\n".join(context_parts),
                "fallback_message": ""
            }
        else:
            # Explicit No-Answer Handling (Mandatory Specification)
            return {
                "found": False,
                "sources": [],
                "context_text": "",
                "fallback_message": "No relevant company information was found in the TechCare knowledge base. Please verify this information manually."
            }
