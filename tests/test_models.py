"""
Unit tests for domain models (Message, ConversationState, CoachingFeedback).
"""

import unittest
from models.conversation import Message, ConversationState, CoachingFeedback


class TestConversationModels(unittest.TestCase):

    def test_message_creation(self):
        msg = Message(speaker="customer", text="I need help with my refund.")
        self.assertEqual(msg.speaker, "customer")
        self.assertEqual(msg.text, "I need help with my refund.")
        self.assertTrue(msg.is_customer())
        self.assertIsNotNone(msg.timestamp)
        self.assertIsNone(msg.image_path)

    def test_agent_message(self):
        msg = Message(speaker="agent", text="I can help you with that.")
        self.assertFalse(msg.is_customer())

    def test_coaching_feedback(self):
        feedback = CoachingFeedback(
            tone_score=9,
            empathy_score=8,
            clarity_score=7,
            coaching_tip="Acknowledge the delay."
        )
        self.assertEqual(feedback.tone_score, 9)
        self.assertEqual(feedback.empathy_score, 8)
        self.assertEqual(feedback.clarity_score, 7)
        self.assertEqual(feedback.coaching_tip, "Acknowledge the delay.")

    def test_conversation_state_lifecycle(self):
        state = ConversationState()
        self.assertEqual(len(state.history), 0)
        self.assertEqual(state.sentiment, "NEUTRAL")

        # Add customer message
        state.add_message("customer", "My order is delayed.")
        self.assertEqual(len(state.history), 1)
        self.assertEqual(state.last_customer_message.text, "My order is delayed.")

        # Add agent message
        state.add_message("agent", "Checking order status.")
        self.assertEqual(len(state.history), 2)
        self.assertEqual(state.last_customer_message.text, "My order is delayed.")

        # Test clear
        state.clear()
        self.assertEqual(len(state.history), 0)
        self.assertIsNone(state.last_customer_message)


if __name__ == "__main__":
    unittest.main()
