"""
Unit tests for Milestone 4 Coaching, Multimodal, and Response Generation services.
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from models.conversation import ConversationState
from services.multimodal_service import validate_image, analyze_image
from services.coaching_service import generate_customer_context_coaching, evaluate_agent_draft
from services.response_service import generate_suggested_response
import config


class TestCoachingAndMultimodal(unittest.TestCase):

    def setUp(self):
        # Create a small dummy image for testing
        self.test_img_path = config.TEMP_UPLOAD_DIR / "test_sample.png"
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 200  # Red tint for LED indicator test
        img = Image.fromarray(img_array)
        img.save(self.test_img_path)

    def tearDown(self):
        if self.test_img_path.exists():
            self.test_img_path.unlink()

    # ==========================================
    # 1. MULTIMODAL & IMAGE VALIDATION TESTS
    # ==========================================
    def test_validate_image_valid(self):
        is_valid, msg = validate_image(str(self.test_img_path))
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Image is valid.")

    def test_validate_image_non_existent(self):
        is_valid, msg = validate_image("non_existent_file.png")
        self.assertFalse(is_valid)
        self.assertIn("not found", msg)

    def test_analyze_image_success(self):
        result = analyze_image(str(self.test_img_path), customer_text="My router is showing a red light")
        self.assertTrue(result["valid"])
        self.assertIn("Hardware Diagnostic", result["detected_element"])
        self.assertGreater(result["confidence"], 0.7)

    # ==========================================
    # 2. CUSTOMER CONTEXT COACHING TESTS
    # ==========================================
    def test_customer_context_coaching_negative_urgency(self):
        state = ConversationState()
        state.sentiment = "NEGATIVE"
        state.urgency = "HIGH"
        state.escalation_risk = "HIGH"
        state.language = "te"
        state.history.append(state.add_message("customer", "నా డబ్బులు తిరిగి ఇవ్వండి!"))

        feedback = generate_customer_context_coaching(state)
        self.assertIsNotNone(feedback.coaching_tip)
        self.assertIn("empathy", feedback.coaching_tip.lower())
        self.assertIn("escalation", feedback.coaching_tip.lower())
        self.assertIn("telugu", feedback.coaching_tip.lower())

    # ==========================================
    # 3. AGENT DRAFT EVALUATION TESTS
    # ==========================================
    def test_evaluate_agent_draft_high_quality(self):
        state = ConversationState()
        state.sentiment = "NEGATIVE"
        agent_draft = "Thank you for contacting Nexora Technologies. I sincerely apologize for the inconvenience. I will process your refund within 24 to 48 hours."
        
        feedback = evaluate_agent_draft(agent_draft, state)
        self.assertGreaterEqual(feedback.tone_score, 8)
        self.assertGreaterEqual(feedback.empathy_score, 8)
        self.assertGreaterEqual(feedback.clarity_score, 8)

    def test_evaluate_agent_draft_poor_quality(self):
        state = ConversationState()
        state.sentiment = "NEGATIVE"
        agent_draft = "It is not our fault. Your refund will come soon, just wait."
        
        feedback = evaluate_agent_draft(agent_draft, state)
        # Tone and empathy should be penalized for defensive phrase and lack of apology
        self.assertLessEqual(feedback.tone_score, 5)
        self.assertLessEqual(feedback.clarity_score, 6)
        self.assertIn("defensive", feedback.coaching_tip.lower())

    # ==========================================
    # 4. SUGGESTED RESPONSE GENERATION TESTS
    # ==========================================
    def test_generate_suggested_response_english(self):
        state = ConversationState()
        state.language = "en"
        state.sentiment = "NEGATIVE"
        state.intent = "Refund Request"
        state.retrieved_sources = [{
            "document": "Nexora Technologies Refund Policy",
            "section": "Duplicate Deduction Procedure",
            "text": "Excess transaction is reversed within 48 hours."
        }]

        response = generate_suggested_response(state)
        self.assertIn("Nexora Technologies Refund Policy", response)
        self.assertIn("apologize", response.lower())

    def test_generate_suggested_response_telugu(self):
        state = ConversationState()
        state.language = "te"
        state.sentiment = "NEGATIVE"
        state.intent = "Technical Issue"
        state.retrieved_sources = [{
            "document": "Nexora Technologies Router Troubleshooting",
            "section": "Step-by-Step Fix for Red Light Issue",
            "text": "Power cycle router."
        }]

        response = generate_suggested_response(state)
        self.assertIn("Nexora Technologies Router Troubleshooting", response)
        self.assertIn("నమస్కారం", response)


if __name__ == "__main__":
    unittest.main()
