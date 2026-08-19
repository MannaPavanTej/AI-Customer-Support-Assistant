"""
Data models representing the core domain of customer support conversations.

Includes:
- Message: An individual message turn.
- CoachingFeedback: Live quality and guidance evaluation for the support agent.
- ConversationState: The full active state of the current customer conversation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Message:
    """
    Represents a single message in the customer-agent dialogue.

    Attributes:
        speaker: 'customer' or 'agent'
        text: The message content
        timestamp: Time the message was recorded
        image_path: Optional path to an uploaded image (e.g. screenshot or device photo)
    """
    speaker: str
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    image_path: Optional[str] = None

    def is_customer(self) -> bool:
        """Returns True if the message was sent by the customer."""
        return self.speaker.lower() == "customer"


@dataclass
class CoachingFeedback:
    """
    Represents AI coaching critique for an agent's drafted or planned response.

    Attributes:
        tone_score: Rating from 1 to 10 (e.g. professional, polite)
        empathy_score: Rating from 1 to 10 (e.g. understanding customer frustration)
        clarity_score: Rating from 1 to 10 (e.g. actionable and easy to understand)
        coaching_tip: Practical suggestion to improve the response
    """
    tone_score: int = 8
    empathy_score: int = 8
    clarity_score: int = 8
    coaching_tip: str = "Acknowledge the customer's request politely and provide clear next steps."


@dataclass
class ConversationState:
    """
    Maintains the state of the active customer-agent conversation session.

    Attributes:
        history: Chronological list of Message objects
        language: Detected language (e.g. 'en', 'hi', 'te')
        sentiment: Overall detected sentiment (e.g. 'POSITIVE', 'NEUTRAL', 'NEGATIVE')
        sentiment_confidence: Confidence score of sentiment prediction (0.0 to 1.0)
        intent: Detected intent category (e.g. 'refund_request', 'technical_issue')
        intent_confidence: Confidence score of intent classification (0.0 to 1.0)
        urgency: Level of urgency ('LOW', 'MEDIUM', 'HIGH')
        escalation_risk: Risk of customer escalation ('LOW', 'MEDIUM', 'HIGH')
        key_issue: Summary of the main problem discussed
        retrieved_sources: Knowledge base chunks retrieved for context
        suggested_response: Grounded draft reply prepared by AI for agent review
        coaching: Current CoachingFeedback object
    """
    history: List[Message] = field(default_factory=list)
    language: str = "en"
    sentiment: str = "NEUTRAL"
    sentiment_confidence: float = 0.0
    intent: str = "General Query"
    intent_confidence: float = 0.0
    urgency: str = "LOW"
    escalation_risk: str = "LOW"
    key_issue: str = "Awaiting customer input..."
    retrieved_sources: List[Dict[str, Any]] = field(default_factory=list)
    suggested_response: str = ""
    coaching: Optional[CoachingFeedback] = None

    def add_message(self, speaker: str, text: str, image_path: Optional[str] = None) -> Message:
        """
        Appends a new message to the conversation history.

        Args:
            speaker: 'customer' or 'agent'
            text: Content of the message
            image_path: Optional file path of an uploaded image

        Returns:
            The created Message object
        """
        msg = Message(speaker=speaker, text=text, image_path=image_path)
        self.history.append(msg)
        return msg

    def clear(self) -> None:
        """Resets the conversation state for a fresh session."""
        self.history.clear()
        self.language = "en"
        self.sentiment = "NEUTRAL"
        self.sentiment_confidence = 0.0
        self.intent = "General Query"
        self.intent_confidence = 0.0
        self.urgency = "LOW"
        self.escalation_risk = "LOW"
        self.key_issue = "Awaiting customer input..."
        self.retrieved_sources.clear()
        self.suggested_response = ""
        self.coaching = None

    @property
    def last_customer_message(self) -> Optional[Message]:
        """Returns the most recent message sent by the customer."""
        for msg in reversed(self.history):
            if msg.is_customer():
                return msg
        return None
