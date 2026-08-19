"""
Unit tests for Milestone 2 Conversation Analysis Services.
Tests Language Detection, Sentiment Analysis, Intent Classification, and Risk Evaluation across English, Hindi, and Telugu.
"""

import unittest
from services.language_service import detect_language
from services.sentiment_service import analyze_sentiment
from services.intent_service import classify_intent
from services.risk_service import evaluate_conversation_risk
from models.conversation import ConversationState


class TestAnalysisServices(unittest.TestCase):

    # ==========================================
    # 1. LANGUAGE DETECTION TESTS
    # ==========================================
    def test_language_detection_english(self):
        code, name, conf = detect_language("I want a refund for my order.")
        self.assertEqual(code, "en")
        self.assertEqual(name, "English")
        self.assertGreater(conf, 0.5)

    def test_language_detection_hindi(self):
        code, name, conf = detect_language("मेरा राउटर काम नहीं कर रहा है।")
        self.assertEqual(code, "hi")
        self.assertIn("Hindi", name)
        self.assertGreater(conf, 0.5)

    def test_language_detection_telugu(self):
        code, name, conf = detect_language("నా రీఫండ్ ఇంకా రాలేదు, దయచేసి చూడండి.")
        self.assertEqual(code, "te")
        self.assertIn("Telugu", name)
        self.assertGreater(conf, 0.5)

    # ==========================================
    # 2. SENTIMENT ANALYSIS TESTS
    # ==========================================
    def test_sentiment_positive_english(self):
        sent, conf, _ = analyze_sentiment("Thank you so much! Your service was great.", "en")
        self.assertEqual(sent, "POSITIVE")
        self.assertGreater(conf, 0.6)

    def test_sentiment_negative_english(self):
        sent, conf, _ = analyze_sentiment("This is the worst service, my payment was deducted twice!", "en")
        self.assertEqual(sent, "NEGATIVE")
        self.assertGreater(conf, 0.6)

    def test_sentiment_negative_telugu(self):
        sent, conf, _ = analyze_sentiment("చాలా చెత్త సర్వీస్, మోసం చేశారు.", "te")
        self.assertEqual(sent, "NEGATIVE")

    def test_sentiment_neutral_hindi(self):
        sent, conf, _ = analyze_sentiment("मुझे डिलीवरी की स्थिति जाननी है।", "hi")
        self.assertEqual(sent, "NEUTRAL")

    # ==========================================
    # 3. INTENT CLASSIFICATION TESTS
    # ==========================================
    def test_intent_refund_request(self):
        intent, conf, _ = classify_intent("I want my money back, please issue a refund.", "en")
        self.assertEqual(intent, "Refund Request")

    def test_intent_technical_issue(self):
        intent, conf, _ = classify_intent("My router is showing a blinking red light and wifi is disconnected.", "en")
        self.assertEqual(intent, "Technical Issue")

    def test_intent_cancellation(self):
        intent, conf, _ = classify_intent("Please cancel my subscription immediately.", "en")
        self.assertEqual(intent, "Cancellation")

    def test_intent_order_delivery(self):
        intent, conf, _ = classify_intent("Where is my package? Can you track my delivery?", "en")
        self.assertEqual(intent, "Order & Delivery Query")

    # ==========================================
    # 4. CONVERSATION RISK EVALUATION TESTS
    # ==========================================
    def test_risk_evaluation_high_urgency_escalation(self):
        state = ConversationState()
        state.sentiment = "NEGATIVE"
        state.intent = "Refund Request"
        state.add_message("customer", "This is the third time! My payment was deducted twice and I will contact consumer forum lawyer!")
        
        urgency, escalation, key_issue = evaluate_conversation_risk(state)
        self.assertEqual(urgency, "HIGH")
        self.assertEqual(escalation, "HIGH")
        self.assertIn("Duplicate payment", key_issue)

    def test_risk_evaluation_low_risk(self):
        state = ConversationState()
        state.sentiment = "POSITIVE"
        state.intent = "General Query"
        state.add_message("customer", "Could you please tell me your working hours?")
        
        urgency, escalation, key_issue = evaluate_conversation_risk(state)
        self.assertEqual(urgency, "LOW")
        self.assertEqual(escalation, "LOW")


if __name__ == "__main__":
    unittest.main()
