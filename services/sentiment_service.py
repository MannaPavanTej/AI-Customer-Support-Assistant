"""
Sentiment Analysis Service supporting English, Hindi, and Telugu.

Analyzes customer tone and emotion to categorize messages into:
- POSITIVE: Happy, grateful, satisfied
- NEUTRAL: Informational, factual query, routine question
- NEGATIVE: Frustrated, angry, disappointed, distressed
"""

from typing import Tuple, Dict, Any
import re

# Domain-specific emotional markers across supported languages
POSITIVE_MARKERS = {
    "en": ["thank", "thanks", "great", "awesome", "good", "excellent", "helpful", "resolved", "appreciate", "perfect", "love"],
    "hi": ["धन्यवाद", "शुक्रिया", "अच्छा", "बढ़िया", "शानदार", "मददगार", "संतुष्ट", "उत्कृष्ट"],
    "te": ["ధన్యవాదాలు", "థాంక్స్", "మంచిది", "బాగుంది", "సహాయం", "సంతోషం", "సరిపోయింది"]
}

NEGATIVE_MARKERS = {
    "en": ["worst", "terrible", "bad", "horrible", "cheat", "scam", "fraud", "angry", "disappointed", "pathetic", "hate", "delay", "not working", "deducted twice", "useless", "issue", "problem", "broken", "fail", "failed", "unhappy"],
    "hi": ["खराब", "बेकार", "धोखा", "घटिया", "समस्या", "परेशान", "काम नहीं कर रहा", "गुस्सा", "रद्द", "विफल", "नाराज"],
    "te": ["బాగాలేదు", "చెత్త", "మోసం", "పనిచేయడం లేదు", "సమస్య", "విసుగు", "కోపం", "ఆలస్యం", "నష్టం", "ఫెయిల్"]
}


def analyze_sentiment(text: str, language: str = "en") -> Tuple[str, float, Dict[str, Any]]:
    """
    Analyzes sentiment of customer message.

    Args:
        text: Customer message text
        language: Detected language code ('en', 'hi', 'te')

    Returns:
        Tuple of (sentiment_label, confidence_score, detailed_metrics)
    """
    if not text or not text.strip():
        return "NEUTRAL", 1.0, {"positive_hits": 0, "negative_hits": 0}

    lower_text = text.lower()
    
    # Check for keyword matches in the detected language + cross-language matches
    pos_words = POSITIVE_MARKERS.get(language, POSITIVE_MARKERS["en"]) + POSITIVE_MARKERS["en"]
    neg_words = NEGATIVE_MARKERS.get(language, NEGATIVE_MARKERS["en"]) + NEGATIVE_MARKERS["en"]

    pos_score = sum(1 for w in set(pos_words) if w in lower_text)
    neg_score = sum(1 for w in set(neg_words) if w in lower_text)

    # Exclamation mark or aggressive punctuation boost
    if "!" in text or "???" in text:
        if neg_score > 0:
            neg_score += 1

    if neg_score > pos_score:
        confidence = min(0.98, 0.65 + (neg_score * 0.10))
        return "NEGATIVE", round(confidence, 2), {"positive_hits": pos_score, "negative_hits": neg_score}
    elif pos_score > neg_score:
        confidence = min(0.98, 0.65 + (pos_score * 0.10))
        return "POSITIVE", round(confidence, 2), {"positive_hits": pos_score, "negative_hits": neg_score}
    else:
        return "NEUTRAL", 0.80, {"positive_hits": pos_score, "negative_hits": neg_score}
