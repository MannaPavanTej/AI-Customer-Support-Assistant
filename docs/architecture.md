# AgentGuide AI by Nexora Technologies — System Architecture

## 1. Executive Summary
**AgentGuide AI** is an intelligent co-pilot by **Nexora Technologies**, designed to assist human customer-support agents.

The system operates on a **Human-in-the-Loop** model: rather than replacing human agents with an autonomous chatbot, the AI analyzes incoming customer messages in real-time, retrieves grounded company policies via Retrieval-Augmented Generation (RAG), scores agent drafts on quality (Tone, Empathy, Clarity), and provides grounded suggested responses.

---

## 2. High-Level System Architecture

```
                    HUMAN SUPPORT AGENT
                           │
                           ▼
                 ┌──────────────────┐
                 │   STREAMLIT UI   │
                 └────────┬─────────┘
                          │
                  Customer Message
                  + Optional Image (EN / HI / TE)
                          │
                          ▼
                   Input Validation
                          │
                          ▼
                   Language Detection
                   (Unicode + N-gram)
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
           Sentiment           Zero-Shot
           Analysis             Intent
        (Tone + Conf)        (8 Categories)
                 │                 │
                 └────────┬────────┘
                          │
                 Conversation Risk Analyzer
                 (Urgency + Escalation + Key Issue)
                          │
                  ┌───────┴────────┐
                  │ Image provided?│
                  └───────┬────────┘
                    Yes ──┴── No
                     │        │
             Multimodal       │
             Diagnostics      │
                     │        │
                     └────┬───┘
                          │
                 FAISS Vector Index (RAG)
                 (Nexora Technologies Policies, FAQs, Manuals)
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          Two-Way Live       Grounded Reply
         AI Coaching          Generation
      (Tone/Empathy/Clarity)
                 │                 │
                 └────────┬────────┘
                          ▼
                 STREAMLIT AGENT DASHBOARD
                          │
                  HUMAN AGENT REVIEW
                 (Edit / Adopt / Send)
                          │
                          ▼
                 Final Customer Response
                          │
                          ▼
                 SQLite Persistent History
```

---

## 3. Core Modules & Responsibilities

### 3.1 Domain Models (`models/conversation.py`)
- `Message`: Represents an individual message turn with timestamp and optional image attachment.
- `CoachingFeedback`: Structured feedback containing Tone ($1\text{-}10$), Empathy ($1\text{-}10$), Clarity ($1\text{-}10$) scores, and actionable coaching tips.
- `ConversationState`: Maintains active session history, sentiment, confidence, intent, urgency, escalation risk, key issue, and retrieved sources.

### 3.2 NLP & Conversation Intelligence (`services/`)
- `language_service.py`: Hybrid script-based (Unicode) and N-gram detection supporting English, Hindi, and Telugu.
- `sentiment_service.py`: Multilingual sentiment classifier outputting `POSITIVE`, `NEUTRAL`, or `NEGATIVE` with calibrated confidence scores.
- `intent_service.py`: Zero-shot intent classification over 8 core support categories (`Refund Request`, `Technical Issue`, `Cancellation`, `Complaint`, `Purchase & Pricing`, `Order & Delivery Query`, `General Query`, `Feedback`).
- `risk_service.py`: Calculates urgency, escalation risk, and extracts key issue summaries using multi-turn conversation dynamics.
- `multimodal_service.py`: Image validation (format whitelist, size limits, PIL integrity) and visual diagnostics (LED indicators, error screenshots).
- `coaching_service.py`: Two-way coaching (Customer-Context Guidance and real-time Agent Draft Critique).
- `response_service.py`: Grounded suggested response generation in English, Hindi, and Telugu.

### 3.3 Retrieval-Augmented Generation (`rag/`)
- `document_loader.py`: Ingests 12 Nexora Technologies markdown documents across Policies, FAQs, and Manuals.
- `chunker.py`: Splits documents by section headers (`##`), preserving document title and section metadata.
- `embeddings.py`: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) generating 384-dimensional unit-normalized embeddings.
- `vector_store.py`: Local vector index using `faiss.IndexFlatIP` (Exact Inner Product = Cosine Similarity).
- `retriever.py`: Uses deterministic query-term coverage (threshold $\ge 0.35$) with cross-lingual keyword mapping and an explicit out-of-scope fallback.

### 3.4 Persistence & History (`database/`)
- `database.py`: Initializes SQLite schema (`sessions` and `conversation_logs` tables).
- `repository.py`: CRUD operations for session management, turn logging, and Markdown transcript export.

---

## 4. Design Principles
1. **Human-in-the-Loop**: The AI is strictly an assistant/co-pilot.
2. **Zero-Hallucination Policy Grounding**: Company policies are never fabricated; missing policies yield explicit disclaimers.
3. **No Over-Engineering**: Clean, modular Python with Streamlit and SQLite rather than heavy distributed architectures.
4. **Independent Unit Testability**: Every module is tested in isolation with 100% test pass rate.
