"""
End-to-End Integration Tests for Nexora Technologies AI Customer Support Assistant.
Tests complete multi-turn conversational workflows across English, Telugu, Hindi, RAG retrieval, Multimodal diagnostics, Coaching, and Persistence.
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np

import config
from models.conversation import ConversationState
from services.language_service import detect_language
from services.sentiment_service import analyze_sentiment
from services.intent_service import classify_intent
from services.risk_service import evaluate_conversation_risk
from services.multimodal_service import analyze_image
from services.coaching_service import generate_customer_context_coaching, evaluate_agent_draft
from services.response_service import generate_suggested_response
from rag.retriever import KnowledgeRetriever
from database.repository import SupportRepository


class TestEndToEndWorkflows(unittest.TestCase):

    def setUp(self):
        self.test_db_path = config.DATABASE_DIR / "test_integration.db"
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        self.repo = SupportRepository(db_path=self.test_db_path)
        self.retriever = KnowledgeRetriever()

        # Create temporary dummy image
        self.test_img_path = config.TEMP_UPLOAD_DIR / "integration_test_img.png"
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_arr[:, :, 0] = 220  # High red channel
        img = Image.fromarray(img_arr)
        img.save(self.test_img_path)

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        if self.test_img_path.exists():
            self.test_img_path.unlink()

    def test_workflow_english_refund_duplicate_payment(self):
        """End-to-End Test: English Negative Refund Scenario with Escalation Risk."""
        state = ConversationState()
        session_id = self.repo.create_session(language="en")

        # 1. Customer Message
        customer_text = "This is the third time contacting you! My card was deducted twice for one router order and I want an immediate refund!"
        state.add_message("customer", customer_text)

        # 2. NLP Pipeline
        lang_code, _, _ = detect_language(customer_text)
        state.language = lang_code
        self.assertEqual(lang_code, "en")

        sentiment, sent_conf, _ = analyze_sentiment(customer_text, lang_code)
        state.sentiment = sentiment
        state.sentiment_confidence = sent_conf
        self.assertEqual(sentiment, "NEGATIVE")

        intent, intent_conf, _ = classify_intent(customer_text, lang_code)
        state.intent = intent
        state.intent_confidence = intent_conf
        self.assertEqual(intent, "Refund Request")

        urgency, escalation_risk, key_issue = evaluate_conversation_risk(state)
        state.urgency = urgency
        state.escalation_risk = escalation_risk
        state.key_issue = key_issue
        self.assertEqual(urgency, "HIGH")
        self.assertEqual(escalation_risk, "HIGH")

        # 3. Grounded RAG Retrieval
        rag_result = self.retriever.retrieve(customer_text, top_k=2)
        state.retrieved_sources = rag_result["sources"]
        self.assertTrue(rag_result["found"])
        self.assertIn("Refund", rag_result["sources"][0]["document"])

        # 4. AI Coaching & Suggested Response
        state.coaching = generate_customer_context_coaching(state)
        self.assertIn("empathy", state.coaching.coaching_tip.lower())

        suggested_reply = generate_suggested_response(state)
        self.assertIn("Nexora Technologies Refund Policy", suggested_reply)
        self.assertIn("apologize", suggested_reply.lower())

        # 5. Agent Review & Evaluation
        agent_draft = "Thank you for reaching out. I sincerely apologize for the duplicate charge. According to our Refund Policy, duplicate deductions are auto-reversed within 24 to 48 hours. Let me track this for you immediately."
        draft_feedback = evaluate_agent_draft(agent_draft, state)
        self.assertGreaterEqual(draft_feedback.tone_score, 8)
        self.assertGreaterEqual(draft_feedback.empathy_score, 8)

        # 6. Final Approval & SQLite Persistence
        state.add_message("agent", agent_draft)
        log_id = self.repo.log_turn(session_id, state, agent_draft)
        self.assertGreater(log_id, 0)

        # 7. Audit Verification
        logs = self.repo.get_session_logs(session_id)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["urgency"], "HIGH")
        self.assertEqual(logs[0]["intent"], "Refund Request")

    def test_workflow_telugu_multimodal_technical_issue(self):
        """End-to-End Test: Telugu Technical Query with Image Attachment."""
        state = ConversationState()
        session_id = self.repo.create_session(language="te")

        # 1. Customer Message + Image
        customer_text = "నా రౌటర్ లో ఎర్ర లైట్ వెలుగుతోంది, ఇంటర్నెట్ రావడం లేదు."
        state.add_message("customer", customer_text, image_path=str(self.test_img_path))

        # 2. Multimodal Analysis
        visual_ctx = analyze_image(str(self.test_img_path), customer_text=customer_text)
        self.assertTrue(visual_ctx["valid"])
        self.assertIn("Hardware Diagnostic", visual_ctx["detected_element"])

        # 3. NLP Analysis
        lang_code, _, _ = detect_language(customer_text)
        state.language = lang_code
        self.assertEqual(lang_code, "te")

        intent, _, _ = classify_intent(customer_text, lang_code)
        state.intent = intent
        self.assertEqual(intent, "Technical Issue")

        # 4. RAG Retrieval & Telugu Response
        rag_result = self.retriever.retrieve(customer_text, top_k=2)
        state.retrieved_sources = rag_result["sources"]
        self.assertTrue(rag_result["found"])

        suggested_reply = generate_suggested_response(state, visual_context=visual_ctx)
        self.assertIn("నమస్కారం", suggested_reply)
        self.assertIn("Nexora Technologies", suggested_reply)

        # 5. Persistent Logging
        log_id = self.repo.log_turn(session_id, state, suggested_reply, image_path=str(self.test_img_path))
        self.assertGreater(log_id, 0)

    def test_workflow_out_of_scope_query(self):
        """End-to-End Test: Out-of-Scope Zero-Hallucination Query."""
        retriever_result = self.retriever.retrieve("What is the recipe for baking chocolate brownies?")
        self.assertFalse(retriever_result["found"])
        self.assertEqual(len(retriever_result["sources"]), 0)
        self.assertIn("No relevant company information was found", retriever_result["fallback_message"])


if __name__ == "__main__":
    unittest.main()
