# AI-Powered Customer Support Assistant with Live Response Guidance

An intelligent, human-in-the-loop AI co-pilot designed to assist human customer support agents during live customer conversations for **TechCare** (a consumer technology company).

---

## 🎯 Key Project Features
- **Human-in-the-Loop Workflow**: The AI provides real-time guidance, grounding, and draft suggestions, while the human support agent reviews, edits, and sends the final response.
- **Multilingual Support**: Supports customer conversations in **English**, **Hindi**, and **Telugu**.
- **Real-Time Conversation Understanding**: Sentiment analysis (with confidence), zero-shot intent classification, urgency scoring, escalation risk indicators, and key issue identification.
- **Multimodal Visual Diagnostics**: Capability to analyze customer-uploaded hardware photos and UI error screenshots.
- **Grounded Knowledge Base (RAG + FAISS)**: Fast similarity search over TechCare policies, FAQs, and manuals with strict source attribution and no-hallucination handling.
- **Two-Way AI Coaching**:
  - *Customer-Context Coaching*: Tips on empathy, clarity, and de-escalation based on customer state.
  - *Agent Draft Critique*: Real-time tone, empathy, and clarity scoring on the agent's drafted message.
- **Persistent Audit Logging**: SQLite database logging conversations, metrics, and agent decisions.

---

## 🏗️ Architecture Flow
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

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
