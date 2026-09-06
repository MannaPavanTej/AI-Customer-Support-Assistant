"""
Unit tests for Milestone 5 SQLite Persistence and SupportRepository.
"""

import unittest
from pathlib import Path
from database.repository import SupportRepository
from database.database import init_db
from models.conversation import ConversationState, CoachingFeedback
import config


class TestDatabasePersistence(unittest.TestCase):

    def setUp(self):
        # Use a temporary test database
        self.test_db_path = config.DATABASE_DIR / "test_support.db"
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        self.repo = SupportRepository(db_path=self.test_db_path)

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_create_and_get_session(self):
        sid = self.repo.create_session(language="te")
        self.assertTrue(sid.startswith("TC-"))

        sessions = self.repo.get_all_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], sid)
        self.assertEqual(sessions[0]["primary_language"], "te")

    def test_log_turn_and_retrieve(self):
        sid = self.repo.create_session(language="en")
        
        state = ConversationState()
        state.language = "en"
        state.sentiment = "NEGATIVE"
        state.sentiment_confidence = 0.92
        state.intent = "Refund Request"
        state.intent_confidence = 0.88
        state.urgency = "HIGH"
        state.escalation_risk = "HIGH"
        state.key_issue = "Duplicate deduction on card checkout"
        state.retrieved_sources = [{
            "document": "Nexora Technologies Refund Policy",
            "section": "Duplicate Deduction Procedure",
            "text": "Excess transaction is reversed in 48h"
        }]
        state.coaching = CoachingFeedback(
            tone_score=9,
            empathy_score=9,
            clarity_score=8,
            coaching_tip="Acknowledge the customer's frustration."
        )
        state.suggested_response = "I apologize for the delay. Your refund is being processed."
        state.add_message("customer", "I want my money back for the double charge!")

        log_id = self.repo.log_turn(
            session_id=sid,
            state=state,
            agent_message="I have processed your refund, you will receive it in 48 hours."
        )
        self.assertGreater(log_id, 0)

        logs = self.repo.get_session_logs(sid)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["customer_message"], "I want my money back for the double charge!")
        self.assertEqual(logs[0]["agent_message"], "I have processed your refund, you will receive it in 48 hours.")
        self.assertEqual(logs[0]["urgency"], "HIGH")
        self.assertEqual(logs[0]["tone_score"], 9)
        self.assertEqual(len(logs[0]["retrieved_sources"]), 1)

    def test_export_markdown(self):
        sid = self.repo.create_session(language="en")
        state = ConversationState()
        state.add_message("customer", "How do I reset my router?")
        state.key_issue = "Router reset procedure"
        
        self.repo.log_turn(
            session_id=sid,
            state=state,
            agent_message="Hold the reset pinhole for 10 seconds."
        )

        md = self.repo.export_session_markdown(sid)
        self.assertIn("Nexora Technologies Support Audit Transcript", md)
        self.assertIn(sid, md)
        self.assertIn("How do I reset my router?", md)

    def test_delete_session(self):
        sid = self.repo.create_session(language="hi")
        self.repo.delete_session(sid)
        sessions = self.repo.get_all_sessions()
        self.assertEqual(len(sessions), 0)


if __name__ == "__main__":
    unittest.main()
