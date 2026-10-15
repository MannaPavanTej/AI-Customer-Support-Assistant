"""Development of AI-Powered Customer Support Assistant with Live Response Guidance."""

import streamlit as st
from pathlib import Path
from html import escape
import time
import uuid
import config
from models.conversation import ConversationState, Message, CoachingFeedback
from services.language_service import detect_language
from services.sentiment_service import analyze_sentiment
from services.intent_service import classify_intent
from services.risk_service import evaluate_conversation_risk
from services.multimodal_service import validate_image, analyze_image
from services.coaching_service import generate_customer_context_coaching, evaluate_agent_draft
from services.response_service import generate_suggested_response
from services.observability import get_logger
from rag import KnowledgeRetriever
from database import SupportRepository


# ==========================================
# 1. APPLICATION & PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deliberately app-controlled appearance: it works consistently even when the
# browser/Streamlit theme differs from the operating-system theme.
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"

theme = {
    "Light": {
        "bg": "#f3f5f7", "surface": "#ffffff", "surface_alt": "#f7f8fa",
        "text": "#202a35", "muted": "#667482", "border": "#d8dee5",
        "accent": "#365f85", "accent_hover": "#294c6d", "sidebar": "#243442",
        "sidebar_text": "#edf1f4", "shadow": "rgba(25, 42, 58, .05)",
        "glow": "rgba(0, 0, 0, 0)", "input": "#ffffff",
        "sidebar_control_bg": "#ffffff", "sidebar_control_text": "#263746",
    },
    "Dark": {
        "bg": "#171d23", "surface": "#202830", "surface_alt": "#252f38",
        "text": "#e8edf1", "muted": "#aab5bf", "border": "#3a4651",
        "accent": "#6d91b2", "accent_hover": "#81a3c2", "sidebar": "#111920",
        "sidebar_text": "#e6ebef", "shadow": "rgba(0, 0, 0, .16)",
        "glow": "rgba(0, 0, 0, 0)", "input": "#1b242c",
        "sidebar_control_bg": "#27333d", "sidebar_control_text": "#eef2f5",
    },
}[st.session_state.ui_theme]

ui_css = """
<style>
:root { --bg: __BG__; --surface: __SURFACE__; --surface-alt: __SURFACE_ALT__; --text: __TEXT__; --muted: __MUTED__; --border: __BORDER__; --accent: __ACCENT__; --accent-hover: __ACCENT_HOVER__; --sidebar: __SIDEBAR__; --sidebar-text: __SIDEBAR_TEXT__; --sidebar-control-bg: __SIDEBAR_CONTROL_BG__; --sidebar-control-text: __SIDEBAR_CONTROL_TEXT__; --shadow: __SHADOW__; --input: __INPUT__; }
html, body, [class*="css"] { font-family: "Segoe UI", Arial, sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid rgba(255,255,255,.12); }
[data-testid="stSidebar"] * { color:var(--sidebar-text); }
[data-testid="stSidebar"] [data-testid="stCode"], [data-testid="stSidebar"] [data-testid="stCode"] pre, [data-testid="stSidebar"] code { background:var(--sidebar-control-bg) !important; color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-testid="stCode"] *, [data-testid="stSidebar"] pre *, [data-testid="stSidebar"] code * { color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div > div { background:var(--sidebar-control-bg) !important; border-color:rgba(255,255,255,.22) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] input, [data-testid="stSidebar"] [data-baseweb="select"] span, [data-testid="stSidebar"] [role="combobox"], [data-testid="stSidebar"] [role="combobox"] *, [data-testid="stSidebar"] input[role="combobox"] { color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:var(--sidebar-control-bg); border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] summary *, [data-testid="stSidebar"] [data-testid="stExpander"] p { color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] div.stDownloadButton > button, [data-testid="stSidebar"] div.stDownloadButton > button * { background:var(--sidebar-control-bg) !important; color:var(--sidebar-control-text) !important; border-color:rgba(255,255,255,.22) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] div.stButton > button { background:rgba(255,255,255,.08) !important; color:#fff !important; border-color:rgba(255,255,255,.2) !important; }
[data-testid="stSidebar"] .stMetric { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.13); border-radius:5px; padding:9px; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color:#bdc8d8 !important; font-size:.68rem; text-transform:uppercase; letter-spacing:.04em; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#fff !important; font-size:1rem; }
.block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1500px; }
.app-header { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:18px 21px; background:var(--surface); color:var(--text); border:1px solid var(--border); border-top:3px solid var(--accent); border-radius:4px; margin-bottom:20px; box-shadow:0 2px 8px var(--shadow); }
.app-header-title { font-size:1.28rem; font-weight:650; letter-spacing:-.02em; display:flex; align-items:center; gap:10px; }
.app-header-subtitle { font-size:.82rem; color:var(--muted); margin-top:4px; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:4px; background:#e8eef5; color:#304760; font-size:.95rem; font-weight:700; }
.status-pill { display:inline-flex; align-items:center; gap:7px; padding:5px 9px; background:#edf6f0; color:#286244; border:1px solid #cfe4d5; border-radius:3px; font-size:.72rem; font-weight:600; }
.status-dot { width:7px; height:7px; background:#43855e; border-radius:50%; }
.pro-card { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:15px; margin-bottom:13px; box-shadow:0 1px 3px var(--shadow); }
.pro-card-header { font-size:.70rem; font-weight:650; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; }
.metrics-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-bottom:11px; }
.metric-box { background:var(--surface); border:1px solid var(--border); border-radius:3px; padding:9px 10px; min-width:0; }
.metric-box-label, .score-tile-title { font-size:.64rem; font-weight:650; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.metric-box-value { font-size:.88rem; font-weight:650; color:var(--text); margin-top:4px; overflow-wrap:anywhere; }
.badge-pill { display:inline-block; padding:3px 7px; border-radius:3px; font-size:.67rem; font-weight:650; line-height:1.2; }
.badge-green { background:#e8f3eb; color:#286244; }.badge-amber { background:#fff4dc; color:#805b13; }.badge-red { background:#fae9e8; color:#963b35; }.badge-blue { background:#e8f0f8; color:#365d83; }.badge-slate { background:#edf0f3; color:#52606d; }
.coaching-banner { background:var(--surface-alt); border:1px solid var(--border); border-left:3px solid #8795a2; border-radius:3px; padding:11px 12px; margin-top:9px; font-size:.82rem; color:var(--text); line-height:1.5; }.coaching-banner-warning { border-left-color:#c28b32; }
.score-container { display:flex; gap:8px; margin-bottom:8px; flex-wrap:wrap; }.score-tile { flex:1; min-width:85px; background:var(--surface); border:1px solid var(--border); border-radius:3px; padding:8px; text-align:center; }.score-tile-num { font-size:1rem; font-weight:650; color:var(--text); margin-top:4px; }
.suggestion-box { background:var(--surface); border:1px solid var(--border); border-radius:3px; padding:14px; font-size:.87rem; color:var(--text); line-height:1.55; white-space:pre-wrap; }
.workspace-empty { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:25px 18px; color:var(--muted); }.workspace-empty strong { display:block; color:var(--text); font-size:.92rem; margin-bottom:5px; }.notice-card { background:#fff7e8; border:1px solid #ead6a9; border-radius:3px; padding:11px 12px; font-size:.81rem; color:#5d4b28; }
[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--border); border-radius:4px; margin-bottom:8px; box-shadow:none; }
.stTextArea textarea, .stTextInput input { background:var(--input) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:4px !important; font-size:.9rem !important; }.stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 2px rgba(72,96,122,.12) !important; }
div.stButton > button { min-height:37px; border-radius:4px; font-weight:600; } div.stButton > button[kind="primary"] { background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; } div.stButton > button[kind="primary"]:hover { background:var(--accent-hover) !important; border-color:var(--accent-hover) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--muted) !important; } [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border); border-radius:4px; }
@media (max-width:760px) { .block-container { padding:1rem .8rem 2rem; }.app-header { padding:15px; align-items:flex-start; flex-direction:column; }.metrics-grid { grid-template-columns:1fr; } }
</style>
"""
for key, value in theme.items():
    ui_css = ui_css.replace(f"__{key.upper()}__", value)
st.markdown(ui_css, unsafe_allow_html=True)


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
logger = get_logger()

config.validate_production_configuration()
if config.AUTH_ENABLED:
    repo.bootstrap_admin(config.ADMIN_USERNAME, config.ADMIN_PASSWORD, config.ADMIN_ROLE)
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if not st.session_state.current_user:
        st.title(config.APP_TITLE)
        st.caption("Sign in to access customer-support records.")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Sign in", type="primary")
        if login:
            user = repo.authenticate(username.strip(), password)
            if user:
                st.session_state.current_user = user
                st.rerun()
            st.error("Invalid username or password.")
        st.stop()
    current_user = st.session_state.current_user
else:
    current_user = {"username": "development", "role": "admin"}

can_manage_conversations = current_user["role"] in {"admin", "agent"}


def remove_expired_uploads() -> None:
    """Remove only application-owned uploads older than the retention window."""
    cutoff = time.time() - (config.MAX_UPLOAD_AGE_HOURS * 60 * 60)
    for upload in config.TEMP_UPLOAD_DIR.glob("upload_*"):
        try:
            if upload.is_file() and upload.stat().st_mtime < cutoff:
                upload.unlink()
        except OSError:
            continue


remove_expired_uploads()


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

if st.session_state.pop("clear_agent_draft", False):
    st.session_state.agent_draft = ""
    st.session_state.agent_composer_textarea = ""

if "visual_context" not in st.session_state:
    st.session_state.visual_context = None

state: ConversationState = st.session_state.state


# ==========================================
# 4. SIDEBAR: AGENT WORKSPACE CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### Nexora Support")
    st.caption("Customer service workspace")
    st.caption(f"Signed in as: {current_user['username']} ({current_user['role']})")
    st.divider()

    st.radio(
        "Appearance",
        options=("Light", "Dark"),
        key="ui_theme",
        horizontal=True,
        help="Choose the workspace appearance that is most comfortable for your environment.",
    )

    st.markdown("**Current case**")
    st.code(st.session_state.session_id, language="text")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Messages", len(state.history))
    with col_s2:
        st.metric("Language", config.SUPPORTED_LANGUAGES.get(state.language, state.language.upper()).split()[0])

    if st.button("New Conversation", use_container_width=True, type="secondary", disabled=not can_manage_conversations):
        st.session_state.session_id = repo.create_session()
        state.clear()
        st.session_state.agent_draft = ""
        st.session_state.visual_context = None
        st.rerun()

    st.divider()

    with st.expander("How this workspace works", expanded=False):
        st.caption("1. Add the customer message or a screenshot.")
        st.caption("2. Review the policy evidence and coaching guidance.")
        st.caption("3. Edit, approve, and send the final reply.")

    st.divider()

    # Past Sessions & SQLite Audit Explorer
    st.markdown("#### Recent cases")
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
                label="Export case transcript",
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
            <span class="brand-mark">N</span><span>Customer Support</span>
        </div>
        <div class="app-header-subtitle">
            Customer care workspace · Agent-reviewed responses
        </div>
    </div>
    <div style="text-align: right;">
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>Workspace ready</span>
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
            Case reference: {st.session_state.session_id}
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
    st.markdown("##### Conversation")
    
    chat_container = st.container(height=360)
    with chat_container:
        if not state.history:
            st.markdown("""
            <div class="workspace-empty">
                <strong>Start a support conversation</strong>
                Add an incoming customer message below. Guidance, sources, and a reviewable draft will appear here automatically.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in state.history:
                avatar = "C" if msg.is_customer() else "A"
                with st.chat_message(name=msg.speaker, avatar=avatar):
                    st.markdown(f"**{msg.speaker.capitalize()}** &nbsp; <span style='font-size:0.75rem; color:#94A3B8;'>{msg.timestamp[11:19]}</span>", unsafe_allow_html=True)
                    st.write(msg.text)
                    if msg.image_path and Path(msg.image_path).exists():
                        st.image(msg.image_path, caption="Customer Attachment", width=220)

    # Customer Simulator Input Box
    st.markdown("##### Customer message")
    with st.form("customer_input_form", clear_on_submit=True):
        customer_text = st.text_area(
            "Customer message (English, Hindi or Telugu)",
            height=75,
            label_visibility="collapsed"
        )
        
        c_upl1, c_upl2 = st.columns([2, 1])
        with c_upl1:
            uploaded_image = st.file_uploader(
                "Add an image (optional)",
                type=config.ALLOWED_IMAGE_TYPES,
                label_visibility="collapsed"
            )
        with c_upl2:
            submit_customer = st.form_submit_button("Analyze message", type="primary", use_container_width=True, disabled=not can_manage_conversations)

        if submit_customer:
            if customer_text.strip() or uploaded_image:
                saved_image_path = None
                if uploaded_image:
                    # Do not trust a client-supplied filename; retain only its
                    # uploader-filtered extension and create a unique name.
                    suffix = Path(uploaded_image.name).suffix.lower()
                    img_filename = f"upload_{uuid.uuid4().hex}{suffix}"
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
    st.markdown("##### Case overview")

    # Intelligence Metrics Card
    sentiment_badge = "badge-green" if state.sentiment == "POSITIVE" else ("badge-red" if state.sentiment == "NEGATIVE" else "badge-slate")
    urgency_badge = "badge-red" if state.urgency == "HIGH" else ("badge-amber" if state.urgency == "MEDIUM" else "badge-green")
    risk_badge = "badge-red" if state.escalation_risk == "HIGH" else ("badge-amber" if state.escalation_risk == "MEDIUM" else "badge-green")
    lang_display = config.SUPPORTED_LANGUAGES.get(state.language, state.language.upper())
    sentiment_confidence = f"Sentiment confidence: {int(state.sentiment_confidence * 100)}%" if state.history else "Awaiting message"

    st.markdown(f"""
    <div class="pro-card">
        <div class="pro-card-header">
            <span>Message assessment</span>
            <span class="badge-pill badge-slate">{sentiment_confidence}</span>
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
            <strong>Identified Issue:</strong> {escape(state.key_issue)}
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
    st.markdown("##### Reference material")
    if state.retrieved_sources:
        for idx, src in enumerate(state.retrieved_sources, 1):
            match_pct = int(src.get('similarity_score', 0) * 100)
            with st.expander(f"{src.get('document')} · {src.get('section')} (Match: {match_pct}%)"):
                st.caption(f"Category: {src.get('category', 'general').capitalize()} | Source: {src.get('source_file')}")
                st.markdown(src.get("text", "No text available"))
    else:
        if state.history:
            st.markdown("""
            <div class="notice-card">
                No specific policy retrieved for this query. Verify information manually before committing.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption("Grounded policies and manuals will appear here automatically upon receiving a query.")

    # Live AI Coaching Section
    composer_text = st.session_state.get("agent_composer_textarea", "").strip() or st.session_state.agent_draft.strip()
    if composer_text:
        active_coaching = evaluate_agent_draft(composer_text, state)
        coaching_title = "Draft feedback"
    else:
        active_coaching = state.coaching or generate_customer_context_coaching(state)
        coaching_title = "Handling guidance"

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
            <strong>Suggested approach:</strong> {active_coaching.coaching_tip}
        </div>
        """, unsafe_allow_html=True)

    # AI Suggested Response
    st.markdown("##### Suggested reply")
    if state.suggested_response:
        st.markdown(f"""
        <div class="suggestion-box">
{state.suggested_response}
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("Use suggested reply", key="use_ai_suggestion", type="secondary", use_container_width=True):
            st.session_state.agent_draft = state.suggested_response
            st.session_state.agent_composer_textarea = state.suggested_response
            st.rerun()
    else:
        st.caption("A suggested reply will appear after a customer message is analyzed.")


# ==========================================
# 7. BOTTOM PANEL: AGENT RESPONSE COMPOSER
# ==========================================
st.divider()
st.markdown("#### Reply to customer")
st.caption("Review and edit your reply before sending.")

agent_text_input = st.text_area(
    "Response Draft:",
    placeholder="Write a reply to the customer…",
    height=95,
    key="agent_composer_textarea",
    label_visibility="collapsed"
)
st.caption(f"{len(agent_text_input.strip())} characters · Replies are recorded in the case history.")

b_col1, b_col2, _ = st.columns([1.5, 1.5, 5])
with b_col1:
    if st.button("Add reply to case", type="primary", use_container_width=True, disabled=not can_manage_conversations):
        if agent_text_input.strip():
            state.add_message("agent", agent_text_input.strip())
            repo.log_turn(
                session_id=st.session_state.session_id,
                state=state,
                agent_message=agent_text_input.strip()
            )
            st.session_state.clear_agent_draft = True
            st.success("Reply added to the case history.")
            st.rerun()
        else:
            st.warning("Please enter a response before sending.")

with b_col2:
    if st.button("Clear Draft", use_container_width=True):
        st.session_state.clear_agent_draft = True
        st.rerun()
