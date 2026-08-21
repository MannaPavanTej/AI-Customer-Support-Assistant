"""
TechCare Live Support Co-Pilot — Professional Human-in-the-Loop Agent Dashboard.
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
from services.multimodal_service import validate_image, analyze_image
from services.coaching_service import generate_customer_context_coaching, evaluate_agent_draft
from services.response_service import generate_suggested_response
from rag import KnowledgeRetriever
from database import SupportRepository


# ==========================================
# 1. APPLICATION & PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=f"{config.COMPANY_NAME} Support Co-Pilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional SaaS Design System & Modern CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .app-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-header-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 2px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    /* Cards & Containers */
    .pro-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .pro-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Intelligence Metric Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .metric-box-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-box-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 2px;
    }

    /* Status Badges */
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: #DCFCE7; color: #15803D; }
    .badge-amber { background: #FEF3C7; color: #B45309; }
    .badge-red { background: #FEE2E2; color: #B91C1C; }
    .badge-blue { background: #DBEAFE; color: #1D4ED8; }
    .badge-slate { background: #F1F5F9; color: #475569; }

    /* Coaching Banner */
    .coaching-banner {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #166534;
        line-height: 1.4;
    }
    .coaching-banner-warning {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #92400E;
    }

    /* Scores Row */
    .score-container {
        display: flex;
        gap: 12px;
        margin-bottom: 8px;
    }
    .score-tile {
        flex: 1;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .score-tile-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .score-tile-num {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Suggested Reply Box */
    .suggestion-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.88rem;
        color: #1E293B;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Buttons override */
    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 6px 16px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 6px;
        font-weight: 500;
    }

    /* Custom Streamlit adjustments */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-size: 0.9rem;
    }
    .stTextArea textarea:focus {
        border-color: #2563EB;
        box-shadow: 0 0 0 1px #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CACHED SERVICES & SINGLETONS
# ==========================================
@st.cache_resource
def get_retriever() -> KnowledgeRetriever:
    """Cached singleton instance of the FAISS Knowledge Retriever."""
    return KnowledgeRetriever()


@st.cache_resource
def get_repository() -> SupportRepository:
    """Cached singleton instance of the SQLite Support Repository."""
    return SupportRepository()


repo = get_repository()
retriever = get_retriever()


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = repo.create_session()

if "state" not in st.session_state:
    st.session_state.state = ConversationState()
    st.session_state.state.coaching = CoachingFeedback(
        tone_score=8,
        empathy_score=8,
        clarity_score=8,
        coaching_tip="Acknowledge customer inquiry promptly with polite and supportive guidance."
    )

if "agent_draft" not in st.session_state:
    st.session_state.agent_draft = ""

if "visual_context" not in st.session_state:
    st.session_state.visual_context = None

state: ConversationState = st.session_state.state


# ==========================================
# 4. SIDEBAR: AGENT WORKSPACE CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### Support Agent Panel")
    st.caption("TechCare Human-in-the-Loop Co-Pilot")
    st.divider()

    st.markdown("**Active Session**")
    st.code(st.session_state.session_id, language="text")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Messages", len(state.history))
    with col_s2:
        st.metric("Language", config.SUPPORTED_LANGUAGES.get(state.language, state.language.upper()).split()[0])

    if st.button("New Conversation", use_container_width=True, type="secondary"):
        st.session_state.session_id = repo.create_session()
        state.clear()
        st.session_state.agent_draft = ""
        st.session_state.visual_context = None
        st.rerun()

    st.divider()

    # Past Sessions & SQLite Audit Explorer
    st.markdown("#### Past Support Tickets")
    past_sessions = repo.get_all_sessions(limit=15)
    if past_sessions:
        session_options = {s["session_id"]: f"{s['session_id']} · {s['created_at'][5:16]}" for s in past_sessions}
        selected_sid = st.selectbox(
            "Select Ticket to Review:",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            label_visibility="collapsed"
        )

        if selected_sid:
            session_logs = repo.get_session_logs(selected_sid)
            with st.expander(f"Audit Log ({len(session_logs)} turns)"):
                if session_logs:
                    for l in session_logs:
                        st.markdown(f"**Customer:** {l['customer_message']}")
                        st.markdown(f"**Agent:** {l['agent_message']}")
                        st.caption(f"Tone: {l['tone_score']}/10 | Empathy: {l['empathy_score']}/10 | Clarity: {l['clarity_score']}/10 | `{l['intent']}`")
                        st.divider()
                else:
                    st.caption("No turns logged for this ticket yet.")

            transcript_md = repo.export_session_markdown(selected_sid)
            st.download_button(
                label="Export Audit Transcript (.md)",
                data=transcript_md,
                file_name=f"transcript_{selected_sid}.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.caption("No past tickets in database.")


# ==========================================
# 5. TOP HEADER BAR
# ==========================================
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-header-title">
            <span>TechCare Live Support Co-Pilot</span>
        </div>
        <div class="app-header-subtitle">
            AI-Assisted Real-Time Guidance Grounded on TechCare Knowledge Base
        </div>
    </div>
    <div style="text-align: right;">
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>Live Co-Pilot Active</span>
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
            Ticket: {st.session_state.session_id}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 6. MAIN WORKSPACE (2 COLUMNS)
# ==========================================
col_chat, col_guidance = st.columns([1.1, 0.9], gap="large")


# ==========================================
# LEFT COLUMN: LIVE CONVERSATION & INPUT
# ==========================================
with col_chat:
    st.markdown("##### Conversation Stream")
    
    chat_container = st.container(height=360)
    with chat_container:
        if not state.history:
            st.info("Conversation thread is empty. Enter customer message below to begin.")
        else:
            for msg in state.history:
                avatar = "👤" if msg.is_customer() else "🎧"
                with st.chat_message(name=msg.speaker, avatar=avatar):
                    st.markdown(f"**{msg.speaker.capitalize()}** &nbsp; <span style='font-size:0.75rem; color:#94A3B8;'>{msg.timestamp[11:19]}</span>", unsafe_allow_html=True)
                    st.write(msg.text)
                    if msg.image_path and Path(msg.image_path).exists():
                        st.image(msg.image_path, caption="Customer Attachment", width=220)

    # Customer Simulator Input Box
    st.markdown("##### Customer Message Simulator")
    with st.form("customer_input_form", clear_on_submit=True):
        customer_text = st.text_area(
            "Customer Message (English / Hindi / Telugu):",
            placeholder="Enter customer message here (e.g., 'My router is showing a red light' or 'నా రీఫండ్ ఇంకా రాలేదు')...",
            height=75,
            label_visibility="collapsed"
        )
        
        c_upl1, c_upl2 = st.columns([2, 1])
        with c_upl1:
            uploaded_image = st.file_uploader(
                "Attach Screenshot or Hardware Photo (Optional):",
                type=config.ALLOWED_IMAGE_TYPES,
                label_visibility="collapsed"
            )
        with c_upl2:
            submit_customer = st.form_submit_button("Receive Customer Message", type="primary", use_container_width=True)

        if submit_customer:
            if customer_text.strip() or uploaded_image:
                saved_image_path = None
                if uploaded_image:
                    img_filename = f"upload_{uploaded_image.name}"
                    img_path = config.TEMP_UPLOAD_DIR / img_filename
                    with open(img_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    saved_image_path = str(img_path)

                # 1. Record customer message
                state.add_message("customer", customer_text.strip(), image_path=saved_image_path)
                
                # 2. NLP Analysis Engine
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

                # 3. Multimodal Image Analysis
                visual_context = None
                if saved_image_path:
                    visual_context = analyze_image(saved_image_path, customer_text=customer_text.strip())
                st.session_state.visual_context = visual_context

                # 4. Grounded RAG Retrieval
                rag_result = retriever.retrieve(customer_text.strip(), top_k=2)
                state.retrieved_sources = rag_result["sources"]

                # 5. Coaching & Grounded Suggested Draft Generation
                state.coaching = generate_customer_context_coaching(state)
                state.suggested_response = generate_suggested_response(state, visual_context=visual_context)
                
                st.rerun()
            else:
                st.warning("Please enter a customer message or attach an image.")


# ==========================================
# RIGHT COLUMN: LIVE AI GUIDANCE & COPILOT
# ==========================================
with col_guidance:
    st.markdown("##### Live AI Response Guidance")

    # Intelligence Metrics Card
    sentiment_badge = "badge-green" if state.sentiment == "POSITIVE" else ("badge-red" if state.sentiment == "NEGATIVE" else "badge-slate")
    urgency_badge = "badge-red" if state.urgency == "HIGH" else ("badge-amber" if state.urgency == "MEDIUM" else "badge-green")
    risk_badge = "badge-red" if state.escalation_risk == "HIGH" else ("badge-amber" if state.escalation_risk == "MEDIUM" else "badge-green")
    lang_display = config.SUPPORTED_LANGUAGES.get(state.language, state.language.upper())

    st.markdown(f"""
    <div class="pro-card">
        <div class="pro-card-header">
            <span>Conversation Intelligence</span>
            <span class="badge-pill badge-slate">Confidence: {int(state.sentiment_confidence * 100) if state.sentiment_confidence else 85}%</span>
        </div>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-box-label">Language</div>
                <div class="metric-box-value">{lang_display}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Sentiment</div>
                <div class="metric-box-value"><span class="badge-pill {sentiment_badge}">{state.sentiment}</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Intent</div>
                <div class="metric-box-value" style="font-size:0.85rem;">{state.intent}</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Urgency</div>
                <div class="metric-box-value"><span class="badge-pill {urgency_badge}">{state.urgency}</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Escalation Risk</div>
                <div class="metric-box-value"><span class="badge-pill {risk_badge}">{state.escalation_risk}</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-box-label">Turn Count</div>
                <div class="metric-box-value">{len(state.history)}</div>
            </div>
        </div>
        <div style="font-size:0.8rem; color:#475569; padding-top:4px;">
            <strong>Identified Issue:</strong> {state.key_issue}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multimodal Diagnostics Card (When Image Provided)
    v_ctx = st.session_state.get("visual_context")
    if v_ctx and v_ctx.get("valid"):
        st.markdown(f"""
        <div class="pro-card" style="border-left: 4px solid #3B82F6;">
            <div class="pro-card-header">
                <span>Visual Diagnostics</span>
                <span class="badge-pill badge-blue">{v_ctx.get('resolution')} · {v_ctx.get('format')}</span>
            </div>
            <div style="font-size:0.85rem; font-weight:600; color:#1E293B; margin-bottom:4px;">
                {v_ctx.get('detected_element')}
            </div>
            <div style="font-size:0.82rem; color:#475569;">
                {v_ctx.get('visual_summary')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Knowledge Base / RAG Section
    st.markdown("##### Grounded Knowledge (TechCare RAG)")
    if state.retrieved_sources:
        for idx, src in enumerate(state.retrieved_sources, 1):
            match_pct = int(src.get('similarity_score', 0) * 100)
            with st.expander(f"{src.get('document')} · {src.get('section')} (Match: {match_pct}%)"):
                st.caption(f"Category: {src.get('category', 'general').capitalize()} | Source: {src.get('source_file')}")
                st.markdown(src.get("text", "No text available"))
    else:
        if state.history:
            st.markdown("""
            <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:10px 12px; font-size:0.82rem; color:#92400E;">
                No specific policy retrieved for this query. Verify information manually before committing.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Grounded policies and manuals will appear here automatically upon receiving a query.")

    # Live AI Coaching Section
    composer_text = st.session_state.get("agent_composer_textarea", "").strip() or st.session_state.agent_draft.strip()
    if composer_text:
        active_coaching = evaluate_agent_draft(composer_text, state)
        coaching_title = "Response Quality Evaluation (Live Draft)"
    else:
        active_coaching = state.coaching or generate_customer_context_coaching(state)
        coaching_title = "Agent Coaching Advice"

    st.markdown(f"##### {coaching_title}")
    if active_coaching:
        tone_cls = "badge-green" if active_coaching.tone_score >= 8 else ("badge-amber" if active_coaching.tone_score >= 6 else "badge-red")
        emp_cls = "badge-green" if active_coaching.empathy_score >= 8 else ("badge-amber" if active_coaching.empathy_score >= 6 else "badge-red")
        clar_cls = "badge-green" if active_coaching.clarity_score >= 8 else ("badge-amber" if active_coaching.clarity_score >= 6 else "badge-red")

        st.markdown(f"""
        <div class="score-container">
            <div class="score-tile">
                <div class="score-tile-title">Tone</div>
                <div class="score-tile-num"><span class="badge-pill {tone_cls}">{active_coaching.tone_score}/10</span></div>
            </div>
            <div class="score-tile">
                <div class="score-tile-title">Empathy</div>
                <div class="score-tile-num"><span class="badge-pill {emp_cls}">{active_coaching.empathy_score}/10</span></div>
            </div>
            <div class="score-tile">
                <div class="score-tile-title">Clarity</div>
                <div class="score-tile-num"><span class="badge-pill {clar_cls}">{active_coaching.clarity_score}/10</span></div>
            </div>
        </div>
        <div class="coaching-banner {'coaching-banner-warning' if ('defensive' in active_coaching.coaching_tip.lower() or 'apolog' in active_coaching.coaching_tip.lower()) else ''}">
            <strong>Guidance:</strong> {active_coaching.coaching_tip}
        </div>
        """, unsafe_allow_html=True)

    # AI Suggested Response
    st.markdown("##### Grounded Suggested Response")
    if state.suggested_response:
        st.markdown(f"""
        <div class="suggestion-box">
{state.suggested_response}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("Use Suggested Draft in Composer", key="use_ai_suggestion", type="secondary", use_container_width=True):
            st.session_state.agent_draft = state.suggested_response
            st.rerun()
    else:
        st.caption("AI-generated draft responses will appear here as the conversation unfolds.")


# ==========================================
# 7. BOTTOM PANEL: AGENT RESPONSE COMPOSER
# ==========================================
st.divider()
st.markdown("#### Agent Response Composer")
st.caption("Review, edit, or customize the drafted response before sending it to the customer.")

agent_text_input = st.text_area(
    "Response Draft:",
    value=st.session_state.agent_draft,
    placeholder="Type response or click 'Use Suggested Draft in Composer' above to populate...",
    height=95,
    key="agent_composer_textarea",
    label_visibility="collapsed"
)

b_col1, b_col2, _ = st.columns([1.8, 1.2, 5])
with b_col1:
    if st.button("Send Response to Customer", type="primary", use_container_width=True):
        if agent_text_input.strip():
            state.add_message("agent", agent_text_input.strip())
            repo.log_turn(
                session_id=st.session_state.session_id,
                state=state,
                agent_message=agent_text_input.strip()
            )
            st.session_state.agent_draft = ""
            st.success("Response sent to customer and logged to database.")
            st.rerun()
        else:
            st.warning("Please enter a response before sending.")

with b_col2:
    if st.button("Clear Draft", use_container_width=True):
        st.session_state.agent_draft = ""
        st.rerun()
