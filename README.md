# AgentGuide AI — Customer Support Intelligence Platform

**AgentGuide AI** is an intelligent, human-in-the-loop co-pilot by **Nexora Technologies**, designed to assist support agents during live customer conversations.

---

## 🌟 Key Features

1. **Human-in-the-Loop Co-Pilot**:
   - The AI acts as an advisor, generating live analytics, grounded knowledge citations, live coaching, and suggested drafts.
   - The human support agent reviews, edits, and approves every outgoing response.
2. **Three-Language Multilingual Support**:
   - Seamless support for **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)** with hybrid Unicode script detection and statistical N-gram fallback.
3. **Real-Time Conversation Understanding**:
   - **Sentiment Analysis**: Emotional polarity (`POSITIVE`, `NEUTRAL`, `NEGATIVE`) with confidence scoring.
   - **Zero-Shot Intent Classification**: Categorizes across 8 core support areas (`Refund Request`, `Technical Issue`, `Cancellation`, `Complaint`, `Purchase & Pricing`, `Order & Delivery Query`, `General Query`, `Feedback`).
   - **Risk Evaluation**: Urgency (`LOW`, `MEDIUM`, `HIGH`), Escalation Risk (`LOW`, `MEDIUM`, `HIGH`), and Key Issue summarization.
4. **Multimodal Visual Diagnostics**:
   - Upload and inspect hardware photos (router warning LED indicators) and checkout error screenshots with strict integrity validation.
5. **Grounded FAISS RAG Retrieval**:
   - Fast vector similarity search over 12 Nexora Technologies policies, FAQs, and manuals.
   - **Grounded-response guardrail**: Enforces cosine-similarity thresholding and a clear manual-verification fallback for out-of-scope queries.
6. **Two-Way Live AI Coaching**:
   - *Customer-Context Guidance*: Upfront coaching on empathy and de-escalation based on customer state.
   - *Agent-Draft Live Critique*: Real-time scoring of Tone ($1\text{-}10$), Empathy ($1\text{-}10$), and Clarity ($1\text{-}10$) as the agent types into the composer.
7. **Persistent SQLite Audit Database**:
   - Automatically logs complete conversation turns, analysis metrics, quality scores, and retrieved sources.
   - Sidebar history explorer with turn review and Markdown transcript export.

---

## 🏗️ System Architecture Flow

```
Customer Message (Text/Image in EN/HI/TE)
               │
               ▼
     Streamlit Agent UI
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Language  Sentiment    Intent
Detection  Analysis  Classification
    └──────────┬──────────┘
               │
     Urgency & Risk Evaluation
               │
     Multimodal Image Analysis (Optional)
               │
     FAISS RAG Knowledge Retrieval
               │
     AI Coaching & Suggested Response
               │
     Human Support Agent Review / Edit
               │
     Approved Response & SQLite History
```

---

## 📁 Project Directory Structure

```
AI_Customer_Coaching/
├── app.py                     # Streamlit Agent Workspace & Dashboard Entry Point
├── config.py                  # Central configuration constants, paths, and thresholds
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # Project overview and setup instructions
│
├── models/
│   ├── __init__.py
│   └── conversation.py        # Message, ConversationState, CoachingFeedback dataclasses
│
├── services/
│   ├── __init__.py
│   ├── language_service.py    # English / Hindi / Telugu script & N-gram LID
│   ├── sentiment_service.py   # Multilingual sentiment analysis & confidence scoring
│   ├── intent_service.py      # Zero-shot intent classification
│   ├── risk_service.py        # Urgency, Escalation Risk, and Key Issue analyzer
│   ├── multimodal_service.py  # Image validation and hardware/UI diagnostics
│   ├── coaching_service.py    # Two-way coaching & real-time draft critique
│   └── response_service.py    # Multilingual grounded draft reply generator
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py     # Ingests markdown files with title/category extraction
│   ├── chunker.py             # Semantic section-based chunking with metadata
│   ├── embeddings.py          # SentenceTransformers (all-MiniLM-L6-v2) 384-d embeddings
│   ├── vector_store.py        # FAISS IndexFlatIP (Cosine Similarity) persistence
│   └── retriever.py           # Similarity search with zero-hallucination guardrail
│
├── database/
│   ├── __init__.py
│   ├── database.py            # SQLite schema (sessions & conversation_logs tables)
│   └── repository.py          # CRUD data access and Markdown export
│
├── knowledge_base/
│   ├── policies/              # Refund, return, cancellation, replacement, warranty
│   ├── faq/                   # Payment, delivery, account, product FAQs
│   └── manuals/               # Router troubleshooting, device setup, connectivity
│
├── vector_store/              # Serialized FAISS index (index.faiss & metadata.pkl)
│
├── tests/
│   ├── test_models.py         # Unit tests for domain models
│   ├── test_analysis.py       # Unit tests for Language, Sentiment, Intent, Risk
│   ├── test_rag.py            # Unit tests for RAG pipeline & no-answer fallback
│   ├── test_coaching.py       # Unit tests for Multimodal, Coaching, Response generator
│   ├── test_database.py       # Unit tests for SQLite persistence & session logs
│   └── test_integration.py    # End-to-End multi-turn workflow integration tests
│
└── docs/
    ├── architecture.md        # Comprehensive system architecture document
    └── viva_preparation.md    # Viva defense Q&A cheat sheet & talking points
```

---

## 🚀 Quickstart Guide

### 1. Set Up Environment
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m unittest discover -s tests -v
```
The suite contains 38 unit and integration tests. It runs fully offline: when the optional MiniLM model has not been cached locally, RAG uses a deterministic token-hash fallback. For stronger semantic retrieval in deployment, cache `all-MiniLM-L6-v2` during image/build setup.

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Sample Demonstration Scenarios

1. **English Payment Issue & Escalation**:
   - Customer Message: `"This is the third time contacting you! My card was deducted twice for one router order and I want an immediate refund!"`
   - AI Guidance: Negative sentiment | Refund Request | HIGH Urgency | HIGH Escalation Risk | Retrieves Nexora Technologies Refund Policy | Pre-populates grounded apology with 24-48h refund timeline.
2. **Telugu Technical Issue with Router Photo**:
   - Customer Message: `"నా రౌటర్ లో ఎర్ర లైట్ వెలుగుతోంది, ఇంటర్నెట్ రావడం లేదు."` + Attach router photo.
   - AI Guidance: Telugu language | Technical Issue | Multimodal Red LED warning detected | Retrieves Router Troubleshooting Manual | Generates polite grounded reply in Telugu.
3. **Hindi Delivery Inquiry**:
   - Customer Message: `"नमस्ते, मुझे मेरे ऑर्डर की डिलीवरी स्थिति जाननी है।"`
   - AI Guidance: Hindi language | Order & Delivery Query | Retrieves Delivery FAQ.
4. **Out-of-Scope Guardrail Query**:
   - Customer Message: `"What is the recipe for baking chocolate cookies?"`
   - AI Guidance: Explicit warning: `⚠️ No relevant company information was found in the knowledge base. Please verify this information manually.`

---

## 🎓 Academic Defense & Viva Resources
Detailed technical explanations, algorithm rationale, and model selection defenses are available in:
- [docs/architecture.md](file:///c:/Users/PAVAN%20TEJ/OneDrive/Desktop/AI_Customer_Coaching/docs/architecture.md)
- [docs/viva_preparation.md](file:///c:/Users/PAVAN%20TEJ/OneDrive/Desktop/AI_Customer_Coaching/docs/viva_preparation.md)

---

## Production Readiness Notes

This repository is a human-in-the-loop support prototype. Do not use it with real customer data until authentication, role-based access, encrypted storage, retention/deletion policies, and centralized monitoring are in place. Uploaded images are stored temporarily and automatically removed after 24 hours. The visual diagnostic feature is a lightweight image heuristic, not a certified device or screenshot recognition model.

The local foundation for those controls is included: set `APP_ENV=production`, `AUTH_ENABLED=true`, administrator credentials, and a Fernet `DATA_ENCRYPTION_KEY` in `.env`. Production startup deliberately stops if any of these settings are missing.
