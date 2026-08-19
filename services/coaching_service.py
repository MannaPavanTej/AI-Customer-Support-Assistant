"""
AI Live Coaching Service.

Provides:
1. Customer-Context Coaching (advice based on customer sentiment, intent, urgency, escalation risk)
2. Agent Draft Response Evaluation (real-time Tone, Empathy, and Clarity scoring with improvement tips)
"""

from typing import Optional
import re
from models.conversation import ConversationState, CoachingFeedback


# Frustration & Low-Empathy Negative Markers in Agent Drafts
DEFENSIVE_PHRASES = [
    "not our fault", "our policy says no", "read the manual", "you should have",
    "calm down", "nothing we can do", "don't blame us", "can't help you"
]

# Polite & Professional Markers
POLITE_PHRASES = [
    "thank you", "please", "happy to help", "assist you", "glad to",
    "have a great day", "certainly", "welcome"
]

# Empathy Markers
EMPATHY_PHRASES = [
    "apologize", "sorry", "understand your frustration", "inconvenience",
    "completely understand", "trouble", "concern", "assure you", "apologies"
]

# Clarity & Actionability Markers
CLARITY_PHRASES = [
    "within", "hours", "days", "step", "first", "next", "reference number",
    "link", "follow these steps", "timeline", "confirm"
]


def generate_customer_context_coaching(state: ConversationState) -> CoachingFeedback:
    """
    Generates coaching advice for the agent based on the customer's emotional and situational context.

    Args:
        state: Current ConversationState

    Returns:
        CoachingFeedback object with default baseline scores and dynamic coaching tip
    """
    tips = []

    # 1. Sentiment-Based Coaching
    if state.sentiment == "NEGATIVE":
        tips.append("Acknowledge the customer's frustration immediately with genuine empathy and avoid defensive explanations.")
    elif state.sentiment == "POSITIVE":
        tips.append("Reinforce customer satisfaction with a warm, appreciative tone.")
    
    # 2. Urgency & Escalation Coaching
    if state.escalation_risk == "HIGH":
        tips.append("High escalation risk detected: Provide a concrete resolution timeline or offer a direct supervisor ticket.")
    elif state.urgency == "HIGH":
        tips.append("High urgency: Keep response concise, prioritize immediate action, and avoid unnecessary small talk.")

    # 3. Multilingual Coaching
    if state.language == "hi":
        tips.append("Customer is communicating in Hindi. Ensure response is clear and supportive in Hindi or simple bilingual phrasing.")
    elif state.language == "te":
        tips.append("Customer is communicating in Telugu. Maintain respectful Telugu phrasing or clear bilingual instructions.")

    # 4. Knowledge Base Grounding Coaching
    if not state.retrieved_sources and state.history:
        tips.append("Query is outside standard documented policies. Verify details manually before committing to timelines.")

    final_tip = " | ".join(tips) if tips else "Maintain a polite, professional, and clear tone with actionable next steps."

    return CoachingFeedback(
        tone_score=8,
        empathy_score=8 if state.sentiment != "NEGATIVE" else 6,
        clarity_score=8,
        coaching_tip=final_tip
    )


def evaluate_agent_draft(agent_text: str, customer_state: Optional[ConversationState] = None) -> CoachingFeedback:
    """
    Evaluates the human agent's drafted message in real-time.

    Args:
        agent_text: The draft message typed by the support agent
        customer_state: Optional context of the customer they are responding to

    Returns:
        CoachingFeedback with calculated Tone (1-10), Empathy (1-10), Clarity (1-10) scores and improvement tip
    """
    if not agent_text or not agent_text.strip():
        return CoachingFeedback(
            tone_score=7,
            empathy_score=7,
            clarity_score=7,
            coaching_tip="Type your response draft to receive real-time coaching feedback."
        )

    text_lower = agent_text.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    # 1. TONE SCORING (1 to 10)
    tone_score = 7
    if any(p in text_lower for p in POLITE_PHRASES):
        tone_score += 2
    if any(p in text_lower for p in DEFENSIVE_PHRASES):
        tone_score -= 4
    if text_lower.isupper() and word_count > 3:  # All-caps shouting
        tone_score -= 3
    tone_score = max(1, min(10, tone_score))

    # 2. EMPATHY SCORING (1 to 10)
    customer_is_negative = customer_state and customer_state.sentiment == "NEGATIVE"
    empathy_score = 7
    has_empathy = any(p in text_lower for p in EMPATHY_PHRASES)

    if has_empathy:
        empathy_score += 2
    elif customer_is_negative:
        empathy_score -= 3  # Penalty for ignoring a frustrated customer's emotion

    if any(p in text_lower for p in DEFENSIVE_PHRASES):
        empathy_score -= 3
    empathy_score = max(1, min(10, empathy_score))

    # 3. CLARITY SCORING (1 to 10)
    clarity_score = 7
    if any(p in text_lower for p in CLARITY_PHRASES):
        clarity_score += 2
    if "soon" in text_lower or "sometime" in text_lower or "later" in text_lower:
        clarity_score -= 2  # Vague timeline penalty
    if word_count < 5:
        clarity_score -= 2  # Too brief
    elif word_count > 80:
        clarity_score -= 1  # Too verbose/cluttered
    clarity_score = max(1, min(10, clarity_score))

    # 4. ACTIONABLE COACHING TIP GENERATION
    tips = []
    if customer_is_negative and not has_empathy:
        tips.append("The customer is frustrated — include an explicit apology or empathetic acknowledgment.")
    if "soon" in text_lower or "later" in text_lower:
        tips.append("Replace vague words like 'soon' with specific business timelines (e.g., 'within 24-48 business hours').")
    if any(p in text_lower for p in DEFENSIVE_PHRASES):
        tips.append("Avoid defensive phrasing; focus on what TechCare can actively do to resolve the problem.")
    if not any(p in text_lower for p in POLITE_PHRASES):
        tips.append("Add a polite greeting or closing statement (e.g., 'Thank you for reaching out').")

    coaching_tip = " | ".join(tips) if tips else "Excellent response draft! Professional, empathetic, and clear."

    return CoachingFeedback(
        tone_score=tone_score,
        empathy_score=empathy_score,
        clarity_score=clarity_score,
        coaching_tip=coaching_tip
    )
