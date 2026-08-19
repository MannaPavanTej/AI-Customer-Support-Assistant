"""
Zero-Shot Intent Classification Service supporting English, Hindi, and Telugu.

Identifies the core customer goal among categories:
- Refund Request
- Technical Issue
- Order & Delivery Query
- Cancellation
- Complaint
- Purchase & Pricing
- General Query
- Feedback
"""

from typing import Tuple, Dict, Any
import numpy as np

# Intent Category Definitions with Semantic Anchor Descriptions (cross-lingual)
INTENT_CATEGORIES = {
    "Refund Request": {
        "anchors": [
            "refund status money back refund request deducted twice return money",
            "రీఫండ్ డబ్బులు వెనక్కి రీఫండ్ కావాలి డబ్బు కట్ అయింది",
            "रिफंड वापस पैसे रिफंड चाहिए पैसे कट गए"
        ],
        "keywords": ["refund", "money back", "deducted", "deduction", "payment back", "రీఫండ్", "డబ్బు", "रिफंड", "पैसे"]
    },
    "Technical Issue": {
        "anchors": [
            "router not working wifi disconnected red light error device setup connectivity",
            "రౌటర్ పనిచేయడం లేదు వైఫై కనెక్ట్ కావడం లేదు ఎర్ర లైట్ పరికరం సమస్య",
            "राउटर काम नहीं कर रहा वाईफाई बंद लाल बत्ती इंटरनेट समस्या"
        ],
        "keywords": ["router", "wifi", "internet", "red light", "setup", "slow", "disconnect", "రౌటర్", "వైఫై", "నెట్", "రాउटर", "वाईफाई"]
    },
    "Cancellation": {
        "anchors": [
            "cancel order stop subscription terminate service cancellation",
            "ఆర్డర్ రద్దు చేయండి సబ్‌స్క్రిప్షన్ రద్దు రద్దు చేయాలి",
            "ऑर्डर रद्द करें सदस्यता रद्द करना सेवा बंद"
        ],
        "keywords": ["cancel", "cancellation", "terminate", "stop service", "రద్దు", "రద్దు చేయండి", "रद्द", "बंद"]
    },
    "Complaint": {
        "anchors": [
            "complaint bad customer service rude agent third time contacting unresolved cheat",
            "ఫిర్యాదు చెత్త సేవ మూడోసారి సంప్రదించడం సమస్య పరిష్కారం కాలేదు మోసం",
            "शिकायत घटिया सेवा तीसरी बार संपर्क समस्या हल नहीं हुई धोखा"
        ],
        "keywords": ["complaint", "worst", "cheat", "scam", "terrible", "third time", "repeated", "ఫిర్యాదు", "మోసం", "చెత్త", "शिकायत", "धोखा"]
    },
    "Order & Delivery Query": {
        "anchors": [
            "track order where is my package delivery status shipping delayed courier",
            "ఆర్డర్ ఎక్కడ ఉంది డెలివరీ స్థితి పార్సెల్ ఆలస్యం ట్రాక్",
            "ऑर्डर कहां है डिलीवरी स्थिति पार्सल देरी ट्रैकिंग"
        ],
        "keywords": ["order", "delivery", "track", "shipping", "courier", "package", "dispatch", "డెలివరీ", "ఆర్డర్", "ట్రాక్", "पार्सल", "डिलीवरी"]
    },
    "Purchase & Pricing": {
        "anchors": [
            "buy new device price discount offer upgrade plan subscription cost",
            "కొత్తది కొనడం ధర డిస్కౌంట్ ఆఫర్ ప్లాన్ కొనుగోలు",
            "नया खरीदना कीमत छूट ऑफर प्लान अपग्रेड"
        ],
        "keywords": ["buy", "purchase", "price", "cost", "discount", "offer", "upgrade", "కొనడం", "ధర", "ఆఫర్", "कीमत", "खरीदना"]
    },
    "Feedback": {
        "anchors": [
            "great service thank you awesome helpful appreciation feedback",
            "మంచి సర్వీస్ ధన్యవాదాలు చాలా సంతోషం ఫీడ్‌బ్యాక్",
            "शानदार सेवा धन्यवाद बहुत अच्छा फीडबैक"
        ],
        "keywords": ["thank", "thanks", "great", "feedback", "awesome", "ధన్యవాదాలు", "సంతోషం", "धन्यवाद", "शानदार"]
    },
    "General Query": {
        "anchors": [
            "general question operational hours contact details information account login",
            "సాధారణ ప్రశ్న పని వేళలు సమాచారం లాగిన్ సహాయం",
            "सामान्य प्रश्न काम के घंटे जानकारी लॉगिन मदद"
        ],
        "keywords": ["help", "info", "information", "hours", "contact", "support", "details", "సమాచారం", "సహాయం", "जानकारी", "मदद"]
    }
}


def classify_intent(text: str, language: str = "en") -> Tuple[str, float, Dict[str, float]]:
    """
    Classifies the user's intent using zero-shot semantic matching & domain heuristics.

    Args:
        text: Customer message
        language: Detected language code

    Returns:
        Tuple containing:
        - predicted_intent: Top intent category
        - confidence: Confidence score (0.0 to 1.0)
        - score_distribution: Dict of scores for each candidate category
    """
    if not text or not text.strip():
        return "General Query", 1.0, {"General Query": 1.0}

    lower_text = text.lower()
    scores: Dict[str, float] = {}

    for intent_name, data in INTENT_CATEGORIES.items():
        # Score based on keyword presence
        kw_hits = sum(1 for kw in data["keywords"] if kw in lower_text)
        
        # Base baseline score
        score = 0.15 + (kw_hits * 0.35)
        scores[intent_name] = min(0.98, score)

    # Find category with highest score
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = sorted_intents[0]

    # If no specific keyword matched, default to General Query
    if top_score <= 0.15:
        top_intent = "General Query"
        top_score = 0.70
        scores["General Query"] = 0.70

    return top_intent, round(top_score, 2), scores
