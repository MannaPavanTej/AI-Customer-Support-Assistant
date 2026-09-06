"""
Conversation Risk & Context Analysis Service.

Computes:
1. Urgency Level (LOW, MEDIUM, HIGH)
2. Escalation Risk (LOW, MEDIUM, HIGH)
3. Key Issue Summary
"""

from typing import Tuple
import re
from models.conversation import ConversationState

# High Urgency Keywords across languages
URGENT_KEYWORDS = [
    "urgent", "immediately", "asap", "emergency", "hurry", "quick", "critical",
    "twice", "deducted twice", "third time", "repeated", "3rd time", "2nd time",
    "త్వరగా", "వెంటనే", "అర్జెంట్", "మూడోసారి",
    "जल्दी", "तुरंत", "आवश्यक", "तीसरी बार"
]

# Escalation Trigger Keywords across languages
ESCALATION_KEYWORDS = [
    "manager", "supervisor", "senior", "escalate", "escalation", "complaint", "legal",
    "lawyer", "court", "consumer forum", "police", "fraud", "cheat", "scam",
    "మేనేజర్", "ఫిర్యాదు", "కోర్టు", "మోసం",
    "मैनेजर", "शिकायत", "अदालत", "धोखा", "कोर्ट"
]


def evaluate_conversation_risk(state: ConversationState) -> Tuple[str, str, str]:
    """
    Evaluates urgency, escalation risk, and extracts key issue from conversation history and state.

    Args:
        state: Current ConversationState object

    Returns:
        Tuple of (urgency_level, escalation_risk_level, key_issue_summary)
    """
    last_msg = state.last_customer_message
    if not last_msg or not last_msg.text.strip():
        return "LOW", "LOW", "Awaiting customer message..."

    text = last_msg.text.lower()
    customer_message_count = sum(1 for m in state.history if m.is_customer())

    # ==========================================
    # 1. URGENCY EVALUATION
    # ==========================================
    urgency = "LOW"
    urgency_points = 0

    if any(k in text for k in URGENT_KEYWORDS):
        urgency_points += 2

    if state.sentiment == "NEGATIVE":
        urgency_points += 1

    if state.intent in ["Refund Request", "Complaint"]:
        urgency_points += 1

    if customer_message_count >= 3:
        urgency_points += 1

    if urgency_points >= 3:
        urgency = "HIGH"
    elif urgency_points >= 1:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    # ==========================================
    # 2. ESCALATION RISK EVALUATION
    # ==========================================
    escalation_risk = "LOW"
    escalation_points = 0

    if any(k in text for k in ESCALATION_KEYWORDS):
        escalation_points += 3

    if state.sentiment == "NEGATIVE" and urgency == "HIGH":
        escalation_points += 2

    if customer_message_count >= 4 and state.sentiment == "NEGATIVE":
        escalation_points += 2

    if escalation_points >= 2:
        escalation_risk = "HIGH"
    elif escalation_points >= 1:
        escalation_risk = "MEDIUM"
    else:
        escalation_risk = "LOW"

    # ==========================================
    # 3. KEY ISSUE EXTRACTION
    # ==========================================
    # Build a clean, human-readable summary
    intent_label = state.intent
    if state.intent == "Refund Request":
        if "twice" in text or "రెండుసార్లు" in text or "दो बार" in text:
            key_issue = "Duplicate payment deduction & refund"
        else:
            key_issue = "Customer requesting transaction refund"
    elif state.intent == "Technical Issue":
        if "router" in text or "రౌటర్" in text or "राउटर" in text:
            key_issue = "Nexora Technologies router connectivity / setup failure"
        elif "red light" in text or "ఎర్ర" in text or "लाल" in text:
            key_issue = "Device showing red indicator light"
        else:
            key_issue = "Hardware/software technical disruption"
    elif state.intent == "Order & Delivery Query":
        key_issue = "Order delivery status and shipment tracking"
    elif state.intent == "Cancellation":
        key_issue = "Subscription or order cancellation request"
    elif state.intent == "Complaint":
        key_issue = "Customer expressing service dissatisfaction"
    elif state.intent == "Purchase & Pricing":
        key_issue = "Inquiry regarding device pricing / new purchase"
    elif state.intent == "Feedback":
        key_issue = "Customer feedback & appreciation"
    else:
        # Fallback to concise snippet
        clean_first_line = last_msg.text.split("\n")[0]
        key_issue = clean_first_line[:60] + "..." if len(clean_first_line) > 60 else clean_first_line

    return urgency, escalation_risk, key_issue
