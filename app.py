"""AgentGuide AI — Nexora Technologies support-agent workspace."""

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
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deliberately app-controlled appearance: it works consistently even when the
# browser/Streamlit theme differs from the operating-system theme.
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"

theme = {
    "Light": {
        "bg": "#f6f8fc", "surface": "#ffffff", "surface_alt": "#f8fafc",
        "text": "#14213d", "muted": "#64748b", "border": "#dce3ef",
        "accent": "#52657f", "accent_hover": "#41546d", "sidebar": "#10172a",
        "sidebar_text": "#e7edff", "shadow": "rgba(15, 23, 42, .08)",
        "glow": "rgba(99, 102, 241, .16)", "input": "#ffffff",
        "sidebar_control_bg": "#f8fafc", "sidebar_control_text": "#0f172a",
    },
    "Dark": {
        "bg": "#0b1120", "surface": "#111a2c", "surface_alt": "#172238",
        "text": "#e8eefc", "muted": "#a7b4cb", "border": "#293753",
        "accent": "#7180a8", "accent_hover": "#8998c2", "sidebar": "#090e1b",
        "sidebar_text": "#e7edff", "shadow": "rgba(0, 0, 0, .28)",
        "glow": "rgba(129, 140, 248, .17)", "input": "#0d1728",
        "sidebar_control_bg": "#1e293b", "sidebar_control_text": "#f8fafc",
    },
}[st.session_state.ui_theme]

ui_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --bg: __BG__; --surface: __SURFACE__; --surface-alt: __SURFACE_ALT__; --text: __TEXT__; --muted: __MUTED__; --border: __BORDER__; --accent: __ACCENT__; --accent-hover: __ACCENT_HOVER__; --sidebar: __SIDEBAR__; --sidebar-text: __SIDEBAR_TEXT__; --sidebar-control-bg: __SIDEBAR_CONTROL_BG__; --sidebar-control-text: __SIDEBAR_CONTROL_TEXT__; --shadow: __SHADOW__; --glow: __GLOW__; --input: __INPUT__; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: radial-gradient(circle at 8% -10%, var(--glow), transparent 31%), radial-gradient(circle at 96% 4%, var(--glow), transparent 25%), var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--sidebar) 0%, #172554 145%); border-right: 1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color: var(--sidebar-text); }
[data-testid="stSidebar"] [data-testid="stCode"], [data-testid="stSidebar"] [data-testid="stCode"] pre, [data-testid="stSidebar"] code { background:var(--sidebar-control-bg) !important; color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-testid="stCode"] *, [data-testid="stSidebar"] pre *, [data-testid="stSidebar"] code * { color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div > div { background:var(--sidebar-control-bg) !important; border-color:rgba(255,255,255,.22) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] input { background:transparent !important; color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [role="combobox"], [data-testid="stSidebar"] [role="combobox"] *, [data-testid="stSidebar"] input[role="combobox"] { color:var(--sidebar-control-text) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:var(--sidebar-control-bg); border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] summary * { color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"], [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] *, [data-testid="stSidebar"] [data-testid="stExpander"] p, [data-testid="stSidebar"] [data-testid="stExpander"] small { color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] div.stDownloadButton > button, [data-testid="stSidebar"] div.stDownloadButton > button * { background:var(--sidebar-control-bg) !important; color:var(--sidebar-control-text) !important; border-color:rgba(255,255,255,.22) !important; -webkit-text-fill-color:var(--sidebar-control-text) !important; }
[data-testid="stSidebar"] div.stButton > button { background:rgba(255,255,255,.10) !important; color:#fff !important; border-color:rgba(191,219,254,.28) !important; }
[data-testid="stSidebar"] div.stButton > button:hover { background:rgba(255,255,255,.17) !important; }
[data-testid="stSidebar"] .stMetric { background: rgba(255,255,255,.07); border: 1px solid rgba(191,219,254,.16); border-radius: 12px; padding: 10px; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p { color: #b6c7ed !important; font-size: .67rem; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff !important; font-size: 1rem; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1460px; }
.app-header { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:22px 24px; background:linear-gradient(120deg,#312e81 0%,#2563eb 55%,#0891b2 100%); color:#fff; border-radius:18px; margin-bottom:22px; box-shadow:0 18px 38px -18px rgba(37,99,235,.72); }
.app-header-title { font-size:1.5rem; font-weight:700; letter-spacing:-.035em; display:flex; align-items:center; gap:11px; }
.app-header-subtitle { font-size:.84rem; color:#dbeafe; margin-top:4px; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:11px; background:rgba(255,255,255,.17); font-size:1.1rem; }
.status-pill { display:inline-flex; align-items:center; gap:7px; padding:6px 11px; background:rgba(16,185,129,.15); color:#d1fae5; border:1px solid rgba(167,243,208,.3); border-radius:999px; font-size:.74rem; font-weight:600; }
.status-dot { width:7px; height:7px; background:#6ee7b7; border-radius:50%; box-shadow:0 0 0 4px rgba(110,231,183,.12); }
.pro-card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:17px; margin-bottom:16px; box-shadow:0 10px 28px -18px var(--shadow); }
.pro-card-header { font-size:.71rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); margin-bottom:13px; display:flex; align-items:center; justify-content:space-between; }
.metrics-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; margin-bottom:12px; }
.metric-box { background:var(--surface-alt); border:1px solid var(--border); border-radius:11px; padding:10px 11px; min-width:0; }
.metric-box-label, .score-tile-title { font-size:.65rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
.metric-box-value { font-size:.93rem; font-weight:700; color:var(--text); margin-top:4px; overflow-wrap:anywhere; }
.badge-pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:.68rem; font-weight:700; line-height:1.2; }
.badge-green { background:#dcfce7; color:#166534; }.badge-amber { background:#fef3c7; color:#92400e; }.badge-red { background:#fee2e2; color:#b91c1c; }.badge-blue { background:#dbeafe; color:#1d4ed8; }.badge-slate { background:#e2e8f0; color:#334155; }
.coaching-banner { background:color-mix(in srgb, #22c55e 12%, var(--surface)); border-left:4px solid #22c55e; border-radius:9px; padding:12px 13px; margin-top:10px; font-size:.84rem; color:var(--text); line-height:1.48; }.coaching-banner-warning { background:color-mix(in srgb, #f59e0b 13%, var(--surface)); border-left-color:#f59e0b; }
.score-container { display:flex; gap:9px; margin-bottom:8px; flex-wrap:wrap; }.score-tile { flex:1; min-width:96px; background:var(--surface-alt); border:1px solid var(--border); border-radius:11px; padding:9px; text-align:center; }.score-tile-num { font-size:1.05rem; font-weight:700; color:var(--text); margin-top:4px; }
.suggestion-box { background:linear-gradient(135deg,color-mix(in srgb, var(--accent) 10%, var(--surface)),var(--surface)); border:1px solid color-mix(in srgb,var(--accent) 32%,var(--border)); border-radius:13px; padding:15px; font-size:.88rem; color:var(--text); line-height:1.55; white-space:pre-wrap; }
.workspace-empty { background:var(--surface); border:1px dashed var(--border); border-radius:14px; padding:28px 20px; text-align:center; color:var(--muted); }.workspace-empty strong { display:block; color:var(--text); font-size:.95rem; margin-bottom:5px; }.notice-card { background:color-mix(in srgb,#f59e0b 13%,var(--surface)); border:1px solid color-mix(in srgb,#f59e0b 35%,var(--border)); border-radius:10px; padding:11px 13px; font-size:.82rem; color:var(--text); }
[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--border); border-radius:14px; margin-bottom:9px; box-shadow:0 5px 16px -13px var(--shadow); }
.stTextArea textarea, .stTextInput input { background:var(--input) !important; color:var(--text) !important; border:1px solid var(--border) !important; border-radius:10px !important; font-size:.9rem !important; }.stTextArea textarea:focus, .stTextInput input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 18%,transparent) !important; }
div.stButton > button { min-height:38px; border-radius:9px; font-weight:600; transition:transform .16s ease, box-shadow .16s ease; } div.stButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 16px -10px var(--shadow); } div.stButton > button[kind="primary"] { background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; } div.stButton > button[kind="primary"]:hover { background:var(--accent-hover) !important; border-color:var(--accent-hover) !important; }
.stMarkdown, .stCaption, p, label { color:var(--text); }.stCaption, [data-testid="stCaptionContainer"] { color:var(--muted) !important; } [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; }
@media (max-width:760px) { .block-container { padding:1rem .8rem 2rem; }.app-header { padding:16px; align-items:flex-start; flex-direction:column; }.metrics-grid { grid-template-columns:1fr; } }
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
    st.markdown("### Support Agent Panel")
    st.caption("Nexora Technologies · Human-in-the-loop support")
    st.caption(f"Signed in as: {current_user['username']} ({current_user['role']})")
    st.divider()

    st.radio(
        "Appearance",
        options=("Light", "Dark"),
        key="ui_theme",
        horizontal=True,
        help="Choose the workspace appearance that is most comfortable for your environment.",
    )

    st.markdown("**Active Session**")
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
            <span class="brand-mark">✦</span><span>AgentGuide AI</span>
        </div>
        <div class="app-header-subtitle">
            Nexora Technologies · AI-assisted guidance with human approval
        </div>
    </div>
    <div style="text-align: right;">
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>Live Co-Pilot Active</span>
        </div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
            Secure workspace · Ticket: {st.session_state.session_id}
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
            st.markdown("""
            <div class="workspace-empty">
                <strong>Start a support conversation</strong>
                Add an incoming customer message below. Guidance, sources, and a reviewable draft will appear here automatically.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in state.history:
                avatar = "👤" if msg.is_customer() else "🎧"
                with st.chat_message(name=msg.speaker, avatar=avatar):
                    st.markdown(f"**{msg.speaker.capitalize()}** &nbsp; <span style='font-size:0.75rem; color:#94A3B8;'>{msg.timestamp[11:19]}</span>", unsafe_allow_html=True)
                    st.write(msg.text)
                    if msg.image_path and Path(msg.image_path).exists():
                        st.image(msg.image_path, caption="Customer Attachment", width=220)

    # Customer Simulator Input Box
    st.markdown("##### Incoming Customer Message")
    with st.form("customer_input_form", clear_on_submit=True):
        customer_text = st.text_area(
            "Customer Message (English / Hindi / Telugu):",
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
            submit_customer = st.form_submit_button("Receive Customer Message", type="primary", use_container_width=True, disabled=not can_manage_conversations)

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
    st.markdown("##### Grounded Knowledge")
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
            st.session_state.agent_composer_textarea = state.suggested_response
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
    placeholder="Type response or click 'Use Suggested Draft in Composer' above to populate...",
    height=95,
    key="agent_composer_textarea",
    label_visibility="collapsed"
)
st.caption(f"{len(agent_text_input.strip())} characters · The customer sees only the reply you approve and send.")

b_col1, b_col2, _ = st.columns([1.5, 1.5, 5])
with b_col1:
    if st.button("Send Response to Customer", type="primary", use_container_width=True, disabled=not can_manage_conversations):
        if agent_text_input.strip():
            state.add_message("agent", agent_text_input.strip())
            repo.log_turn(
                session_id=st.session_state.session_id,
                state=state,
                agent_message=agent_text_input.strip()
            )
            st.session_state.clear_agent_draft = True
            st.success("Response sent to customer and logged to database.")
            st.rerun()
        else:
            st.warning("Please enter a response before sending.")

with b_col2:
    if st.button("Clear Draft", use_container_width=True):
        st.session_state.clear_agent_draft = True
        st.rerun()
