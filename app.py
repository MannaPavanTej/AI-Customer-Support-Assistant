"""
TechCare Live Support Co-Pilot - Human-in-the-Loop Agent Dashboard.
Entry point for the Streamlit web application.
"""

import streamlit as st
from pathlib import Path
import config
from models.conversation import ConversationState, Message, CoachingFeedback
from services.language_service import detect_language
from services.sentiment_service import analyze_sentiment
from services.intent_service import classify_intent
from services.risk_service import evaluate_conversation_risk


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS for clean presentation
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #212529;
    }
    .coaching-card {
        background-color: #eef7ff;
        border-left: 4px solid #0d6efd;
        border-radius: 4px;
        padding: 12px;
        margin-top: 10px;
    }
    .badge-urgent {
        color: #dc3545;
        font-weight: bold;
    }
    .badge-normal {
        color: #198754;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# 2. SESSION STATE INITIALIZATION
if "state" not in st.session_state:
    st.session_state.state = ConversationState()
    # Initialize with a default coaching card for demonstration
    st.session_state.state.coaching = CoachingFeedback(
        tone_score=8,
        empathy_score=7,
        clarity_score=9,
        coaching_tip="Acknowledge the customer's request and provide a clear, supportive timeline."
    )

if "agent_draft" not in st.session_state:
    st.session_state.agent_draft = ""

state: ConversationState = st.session_state.state


# 3. SIDEBAR: SESSION CONTROLS & INFO
with st.sidebar:
    st.title("🎧 TechCare Agent Hub")
    st.caption("Human-in-the-Loop Support Co-Pilot")
    st.divider()

    st.subheader("Session Information")
    st.write(f"**Total Messages:** {len(state.history)}")
    st.write(f"**Primary Language:** {config.SUPPORTED_LANGUAGES.get(state.language, 'English')}")
    st.write(f"**Company:** {config.COMPANY_NAME}")
    
    st.divider()
    if st.button("🔄 Start New Conversation", use_container_width=True, type="secondary"):
        state.clear()
        st.session_state.agent_draft = ""
        st.rerun()

    st.divider()
    st.info("💡 **Architectural Principle:** The AI acts as an advisor. The human agent remains in full control of all customer communications.")


# 4. MAIN APPLICATION HEADER
st.title(f"🎧 {config.APP_TITLE}")
st.caption(f"{config.APP_SUBTITLE} | Grounded on {config.COMPANY_NAME} Knowledge Base")
st.divider()


# 5. TWO-COLUMN WORKSPACE
col_chat, col_guidance = st.columns([1.1, 0.9], gap="large")


# ==========================================
# LEFT COLUMN: LIVE CONVERSATION STREAM
# ==========================================
with col_chat:
    st.subheader("💬 Live Conversation")
    
    chat_container = st.container(height=380)
    with chat_container:
        if not state.history:
            st.info("No messages in this session yet. Simulate a customer message below to begin.")
        else:
            for msg in state.history:
                avatar = "👤" if msg.is_customer() else "🎧"
                with st.chat_message(name=msg.speaker, avatar=avatar):
                    st.markdown(f"**{msg.speaker.capitalize()}** `[{msg.timestamp}]`")
                    st.write(msg.text)
                    if msg.image_path:
                        st.image(msg.image_path, caption="Customer Attachment", width=250)

    st.divider()

    # Customer Message Simulation Box
    st.markdown("#### 📥 Customer Input Simulator")
    with st.form("customer_input_form", clear_on_submit=True):
        customer_text = st.text_area(
            "Customer Message (English / Hindi / Telugu):",
            placeholder="Type customer message here (e.g., 'My internet router is showing a red light' or 'నా రీఫండ్ ఇంకా రాలేదు')...",
            height=80
        )
        uploaded_image = st.file_uploader(
            "Attach Customer Image (Optional - JPG, PNG):",
            type=config.ALLOWED_IMAGE_TYPES
        )
        submit_customer = st.form_submit_button("📩 Receive Customer Message", type="primary", use_container_width=True)

        if submit_customer:
            if customer_text.strip() or uploaded_image:
                saved_image_path = None
                if uploaded_image:
                    img_filename = f"upload_{uploaded_image.name}"
                    img_path = config.TEMP_UPLOAD_DIR / img_filename
                    with open(img_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    saved_image_path = str(img_path)

                # 1. Add customer message to conversation state
                state.add_message("customer", customer_text.strip(), image_path=saved_image_path)
                
                # 2. Multilingual & NLP Analysis Engine (Milestone 2)
                lang_code, lang_name, lang_conf = detect_language(customer_text.strip())
                state.language = lang_code
                
                sentiment_label, sent_conf, _ = analyze_sentiment(customer_text.strip(), language=lang_code)
                state.sentiment = sentiment_label
                state.sentiment_confidence = sent_conf
                
                intent_label, intent_conf, _ = classify_intent(customer_text.strip(), language=lang_code)
                state.intent = intent_label
                state.intent_confidence = intent_conf
                
                urgency, escalation_risk, key_issue = evaluate_conversation_risk(state)
                state.urgency = urgency
                state.escalation_risk = escalation_risk
                state.key_issue = key_issue

                # Context-aware default response draft
                if state.sentiment == "NEGATIVE":
                    state.suggested_response = f"I am truly sorry for the inconvenience this has caused. I am actively looking into your {state.intent.lower()} right now to ensure this is resolved immediately."
                else:
                    state.suggested_response = f"Thank you for contacting {config.COMPANY_NAME}. I will be happy to assist you with your {state.intent.lower()}."
                
                st.rerun()
            else:
                st.warning("Please enter a message or upload an image.")


# ==========================================
# RIGHT COLUMN: LIVE AI GUIDANCE PANEL
# ==========================================
with col_guidance:
    st.subheader("⚡ Live AI Response Guidance")

    # Metrics Section: Language, Sentiment, Intent, Urgency, Escalation
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Language</div>
            <div class="metric-value">{config.SUPPORTED_LANGUAGES.get(state.language, state.language.upper())}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sentiment</div>
            <div class="metric-value">{state.sentiment}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Intent</div>
            <div class="metric-value">{state.intent}</div>
        </div>
        """, unsafe_allow_html=True)

    m_col4, m_col5, m_col6 = st.columns(3)
    with m_col4:
        urgency_class = "badge-urgent" if state.urgency == "HIGH" else "badge-normal"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Urgency</div>
            <div class="metric-value {urgency_class}">{state.urgency}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        risk_class = "badge-urgent" if state.escalation_risk == "HIGH" else "badge-normal"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Escalation Risk</div>
            <div class="metric-value {risk_class}">{state.escalation_risk}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Confidence</div>
            <div class="metric-value">{int(state.sentiment_confidence * 100) if state.sentiment_confidence else 85}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Key Issue Card
    st.markdown(f"**📌 Identified Key Issue:** `{state.key_issue}`")

    st.divider()

    # Knowledge Base / RAG Section
    st.markdown("#### 📚 Grounded Knowledge Base (TechCare RAG)")
    if state.retrieved_sources:
        for idx, src in enumerate(state.retrieved_sources, 1):
            with st.expander(f"Source {idx}: {src.get('document', 'Knowledge Doc')}"):
                st.write(src.get("text", "No text available"))
    else:
        st.caption("No specific policy retrieved yet. RAG retrieval will automatically activate when a customer query is processed.")

    st.divider()

    # Coaching Section
    st.markdown("#### 💡 AI Live Coaching")
    if state.coaching:
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Tone", f"{state.coaching.tone_score}/10")
        c_col2.metric("Empathy", f"{state.coaching.empathy_score}/10")
        c_col3.metric("Clarity", f"{state.coaching.clarity_score}/10")
        st.markdown(f"""
        <div class="coaching-card">
            <strong>Coaching Tip:</strong> {state.coaching.coaching_tip}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # AI Suggested Response
    st.markdown("#### 🤖 AI Suggested Response (Draft)")
    if state.suggested_response:
        st.info(state.suggested_response)
        if st.button("📋 Use AI Suggestion in Composer", key="use_ai_suggestion"):
            st.session_state.agent_draft = state.suggested_response
            st.rerun()
    else:
        st.caption("Suggested responses will appear here as the conversation unfolds.")


# ==========================================
# BOTTOM PANEL: AGENT RESPONSE COMPOSER
# ==========================================
st.divider()
st.subheader("✍️ Human Agent Response Composer")
st.caption("Review, edit, or customize the response before sending it to the customer.")

agent_text_input = st.text_area(
    "Agent Response Draft:",
    value=st.session_state.agent_draft,
    placeholder="Type your response or click 'Use AI Suggestion' above to populate this box...",
    height=100,
    key="agent_composer_textarea"
)

btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 5])
with btn_col1:
    if st.button("🚀 Send Response to Customer", type="primary", use_container_width=True):
        if agent_text_input.strip():
            state.add_message("agent", agent_text_input.strip())
            st.session_state.agent_draft = ""
            st.success("Response sent to customer successfully.")
            st.rerun()
        else:
            st.warning("Cannot send an empty response.")

with btn_col2:
    if st.button("🧹 Clear Draft", use_container_width=True):
        st.session_state.agent_draft = ""
        st.rerun()
