"""
Grounded Suggested Response Generation Service.

Synthesizes:
- Customer intent & sentiment
- Detected language (English, Hindi, Telugu)
- Grounded TechCare knowledge base retrieved context
- Visual diagnostics (if image was provided)
"""

from typing import Optional, Dict, Any
from models.conversation import ConversationState
import config


def generate_suggested_response(
    state: ConversationState,
    visual_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs a grounded, high-quality draft response for human agent review.

    Args:
        state: Active ConversationState with RAG sources and NLP analytics
        visual_context: Optional visual diagnostics dict from multimodal analysis

    Returns:
        String containing the pre-formulated suggested reply
    """
    lang = state.language
    intent = state.intent
    sentiment = state.sentiment
    sources = state.retrieved_sources

    # ==========================================
    # 1. HINDI RESPONSE GENERATION
    # ==========================================
    if lang == "hi":
        if sentiment == "NEGATIVE":
            greeting = "नमस्ते, हुई असुविधा के लिए हमें गहरा खेद है।"
        else:
            greeting = f"नमस्ते, {config.COMPANY_NAME} कस्टमर सपोर्ट में आपका स्वागत है।"

        if visual_context and visual_context.get("valid"):
            diag = "हम आपके द्वारा भेजे गए स्क्रीनशॉट/फोटो की जांच कर रहे हैं।"
        else:
            diag = ""

        if sources:
            top_src = sources[0]
            body = (
                f"{greeting} {diag}\n\n"
                f"TechCare {top_src['document']} के अनुसार:\n"
                f"हम आपके {intent} को जल्द से जल्द हल करने में आपकी सहायता करेंगे। "
                f"कृपया आवश्यक विवरण साझा करें ताकि हम तुरंत आगे की कार्रवाई कर सकें।"
            )
        else:
            body = (
                f"{greeting} {diag}\n\n"
                f"मैं आपकी समस्या को समझ रहा हूँ और इसे तुरंत हल करने के लिए विवरणों की जांच कर रहा हूँ।"
            )
        return body.strip()

    # ==========================================
    # 2. TELUGU RESPONSE GENERATION
    # ==========================================
    if lang == "te":
        if sentiment == "NEGATIVE":
            greeting = "నమస్కారం, మీకు కలిగిన అసౌకర్యానికి మేము చింతిస్తున్నాము."
        else:
            greeting = f"నమస్కారం, {config.COMPANY_NAME} కస్టమర్ సపోర్ట్‌కు స్వాగతం."

        if visual_context and visual_context.get("valid"):
            diag = "మీరు పంపిన ఫోటో/స్క్రీన్‌షాట్‌ను మేము పరిశీలించాము."
        else:
            diag = ""

        if sources:
            top_src = sources[0]
            body = (
                f"{greeting} {diag}\n\n"
                f"TechCare {top_src['document']} ప్రకారం:\n"
                f"మీ సమస్యను వీలైనంత త్వరగా పరిష్కరించడానికి మేము సిద్ధంగా ఉన్నాము. "
                f"దయచేసి సంబంధిత వివరాలు తెలపండి, మేము వెంటనే సహాయం చేస్తాము."
            )
        else:
            body = (
                f"{greeting} {diag}\n\n"
                f"మీ అభ్యర్థనను మేము పరిశీలిస్తున్నాము, త్వరలోనే సరైన పరిష్కారం అందిస్తాము."
            )
        return body.strip()

    # ==========================================
    # 3. ENGLISH RESPONSE GENERATION (DEFAULT)
    # ==========================================
    if sentiment == "NEGATIVE":
        opening = f"Thank you for reaching out to {config.COMPANY_NAME}. I sincerely apologize for the frustration and inconvenience this situation has caused."
    else:
        opening = f"Thank you for contacting {config.COMPANY_NAME} support. I would be happy to assist you today."

    visual_note = ""
    if visual_context and visual_context.get("valid"):
        visual_note = f"\n\nI have reviewed the attached image ({visual_context.get('detected_element')})."

    if sources:
        top_src = sources[0]
        policy_note = (
            f"\n\nAccording to the TechCare **{top_src['document']}** (*{top_src['section']}*):\n"
            f"We are actively working on resolving your **{intent}**. "
        )
        if "Refund" in top_src["document"] or "Duplicate" in top_src["section"]:
            action_step = "If this is a duplicate deduction, our payment gateway auto-reconciles excess charges within 24 to 48 business hours. If you have your 12-digit UTR/reference number, please share it here so I can verify the transaction immediately."
        elif "Router" in top_src["document"] or "Troubleshooting" in top_src["document"]:
            action_step = "Please try power cycling your router: unplug the power adapter for 30 seconds, reconnect, and allow 90 seconds to reboot. If the red indicator persists, let me know and we will proceed with a reset."
        elif "Return" in top_src["document"]:
            action_step = "Our standard return window is 14 days from delivery. We can arrange a free doorstep pickup once your device serial number is verified."
        else:
            action_step = "Please confirm your registered account email or order ID so I can proceed with the resolution immediately."

        response = f"{opening}{visual_note}{policy_note}{action_step}\n\nPlease let me know if you have any questions!"
    else:
        response = (
            f"{opening}{visual_note}\n\n"
            f"I am actively checking your query regarding **{intent}**. "
            f"Could you please provide your order ID or registered contact details so I can look up your record and provide accurate guidance?"
        )

    return response
