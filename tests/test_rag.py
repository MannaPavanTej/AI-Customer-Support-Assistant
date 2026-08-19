"""
Unit tests for Milestone 3 RAG Pipeline.
Tests Document Loader, Chunker, Embedding Generation, FAISS Vector Store, and Grounded Retriever with No-Answer Fallback.
"""

import unittest
from rag.document_loader import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import compute_embeddings
from rag.vector_store import FAISSVectorStore
from rag.retriever import KnowledgeRetriever, RELEVANCE_THRESHOLD


class TestRAGPipeline(unittest.TestCase):

    def test_document_loader(self):
        docs = load_documents()
        self.assertGreaterEqual(len(docs), 12, "Should load at least 12 knowledge base documents")
        categories = {doc.category for doc in docs}
        self.assertIn("policies", categories)
        self.assertIn("faq", categories)
        self.assertIn("manuals", categories)

    def test_chunker(self):
        docs = load_documents()
        chunks = chunk_documents(docs)
        self.assertGreater(len(chunks), len(docs), "Chunking should produce multiple chunks per document")
        sample_chunk = chunks[0]
        self.assertTrue(hasattr(sample_chunk, "chunk_id"))
        self.assertTrue(hasattr(sample_chunk, "section_title"))
        self.assertTrue(hasattr(sample_chunk, "text"))

    def test_embeddings_generation(self):
        sample_text = "TechCare refund policy applies to duplicate deductions."
        vec = compute_embeddings(sample_text)
        self.assertEqual(len(vec.shape), 2)
        self.assertEqual(vec.shape[1], 384, "Sentence-transformers MiniLM produces 384-dimensional embeddings")

    def test_retriever_relevant_query(self):
        retriever = KnowledgeRetriever()
        result = retriever.retrieve("My account was charged twice for one order, how do I get a refund?")
        self.assertTrue(result["found"])
        self.assertGreater(len(result["sources"]), 0)
        
        top_source = result["sources"][0]
        self.assertIn("Refund", top_source["document"])
        self.assertGreaterEqual(top_source["similarity_score"], RELEVANCE_THRESHOLD)
        self.assertTrue(len(top_source["text"]) > 0)

    def test_retriever_technical_query(self):
        retriever = KnowledgeRetriever()
        result = retriever.retrieve("Why is my router showing a red light and how to reset?")
        self.assertTrue(result["found"])
        self.assertGreater(len(result["sources"]), 0)
        
        top_source = result["sources"][0]
        self.assertTrue("Router" in top_source["document"] or "Troubleshooting" in top_source["section"])

    def test_retriever_out_of_scope_no_answer(self):
        retriever = KnowledgeRetriever()
        result = retriever.retrieve("How do I bake a chocolate brownie cake with strawberries?")
        # Out-of-scope query should not hallucinate company policies
        self.assertFalse(result["found"])
        self.assertEqual(len(result["sources"]), 0)
        self.assertIn("No relevant company information was found", result["fallback_message"])


if __name__ == "__main__":
    unittest.main()
