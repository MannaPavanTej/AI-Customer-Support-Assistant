# Development of AI-Powered Customer Support Assistant with Live Response Guidance

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
[![Tests](https://github.com/MannaPavanTej/AI-Customer-Support-Assistant/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MannaPavanTej/AI-Customer-Support-Assistant/actions/workflows/tests.yml)

**A human-guided customer support workspace by Nexora Technologies.** The AI assistant analyzes support messages, surfaces relevant company guidance, and drafts coaching and response suggestions. A support agent reviews and approves every reply.

## What it does

- **Understands customer messages:** estimates language, sentiment, intent, urgency, and escalation risk.
- **Supports three languages:** English, Hindi (हिंदी), and Telugu (తెలుగు), with script-aware detection and language-specific response templates.
- **Finds company guidance:** searches the included policies, FAQs, and manuals using deterministic term matching and Telugu/Hindi query expansion. Low-relevance queries receive an explicit no-answer fallback.
- **Reviews images:** validates uploaded images and provides lightweight visual diagnostics for device photos and checkout screenshots.
- **Coaches support agents:** gives customer-context coaching and scores draft replies for tone, empathy, and clarity.
- **Keeps a conversation history:** stores sessions and turn details in SQLite and can export transcripts as Markdown.

> This project is an assistive prototype. It prepares suggestions; a human agent remains responsible for reviewing, editing, and sending customer replies.

## See the workflow

```
Customer message + optional image
                 |
                 v
       Streamlit support workspace
                 |
   +-------------+-------------+----------------+
   |             |             |                |
   v             v             v                v
Language,     Urgency and   Knowledge-base   Optional image
sentiment,    escalation    retrieval        diagnostics
and intent    assessment
   |             |             |                |
   +-------------+-------------+----------------+
                 |
                 v
       Coaching + response draft
                 |
                 v
       Agent reviews and edits
                 |
                 v
     Approved reply + SQLite history
```

## Quickstart

### Requirements

- Python 3.11
- Windows, macOS, or Linux

### 1. Get the project and create an environment

From the project directory, create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some machine-learning dependencies are large, so the first installation may take a few minutes.

### 3. Start the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Stop the local server with `Ctrl+C` in the terminal.

## Run the test suite

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same command on pushes and pull requests. The workflow is defined in [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Try these example messages

| Scenario | Example | Expected guidance |
| --- | --- | --- |
| Duplicate payment | “My card was deducted twice for one order and I want a refund.” | Refund guidance, risk assessment, and a response draft |
| Router issue in Telugu | “నా రౌటర్ లో ఎర్ర లైట్ వెలుగుతోంది, ఇంటర్నెట్ రావడం లేదు.” | Telugu analysis and router troubleshooting guidance |
| Delivery question in Hindi | “नमस्ते, मुझे मेरे ऑर्डर की डिलीवरी स्थिति जाननी है।” | Hindi analysis and delivery FAQ guidance |
| Unrelated question | “What is the recipe for chocolate brownies?” | A no-relevant-information fallback |

## How retrieval works

The default retriever splits the local Markdown knowledge base into sections, expands supported Hindi and Telugu terms into English equivalents, and ranks passages by query-term coverage. It returns sources only when the best passage meets the configured `0.35` relevance threshold. Questions with insufficient overlap receive a manual-verification fallback instead of company-specific guidance.

The repository also contains embedding and FAISS utilities, but the current `KnowledgeRetriever` path uses deterministic term matching. This keeps the default retrieval behavior predictable in local development and CI without requiring a downloaded embedding model.

## Project layout

```text
.
├── app.py                  # Streamlit support workspace
├── config.py               # App settings, paths, and limits
├── requirements.txt        # Python dependencies
├── services/               # Language, sentiment, risk, coaching, and response logic
├── rag/                    # Knowledge loading, chunking, retrieval, and vector utilities
├── knowledge_base/         # Policies, FAQs, and product manuals
├── models/                 # Conversation and feedback data models
├── database/               # SQLite persistence and transcript export
├── tests/                  # Unit and integration tests
└── docs/                   # Architecture notes and project Q&A
```

## Configuration and data

The app runs locally with development defaults and does not require API credentials. Optional environment settings are documented in [`.env.example`](.env.example). Copy it to `.env` only when you need to configure authentication or production safeguards; keep populated secrets out of Git.

Conversation history is stored in the local SQLite database. Uploaded images are kept temporarily and the app is configured to remove them after 24 hours. Review the retention and access controls before using real customer information.

## Scope and limitations

- Sentiment and intent are lightweight, rule-based analyses; their scores are guidance, not calibrated probabilities.
- Image diagnostics use basic image properties and heuristics. They are not a general-purpose vision model or a certified hardware diagnosis.
- Retrieval is limited to the knowledge-base content included in this repository. Keep policies and FAQs current, and verify advice before sending it to a customer.
- This repository is a prototype, not a production-ready customer data system. Before production use, configure authentication, access controls, encryption keys, data retention, monitoring, and operational review.

## Project documentation

- [Architecture](docs/architecture.md)
- [Viva preparation and design Q&A](docs/viva_preparation.md)
