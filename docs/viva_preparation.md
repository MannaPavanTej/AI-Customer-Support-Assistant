# Viva & Technical Defense Preparation Guide

## 🎯 1. 30-Second Project Elevator Pitch
> *"AgentGuide AI by Nexora Technologies is an AI-powered customer-support assistant designed to provide live response guidance to human support agents. The system detects the customer's language across English, Hindi, and Telugu, analyzes sentiment and zero-shot intent, and evaluates urgency and escalation risk. It optionally understands customer-uploaded hardware photos and screenshots using multimodal diagnostics. It then uses RAG with a FAISS vector database to retrieve grounded Nexora Technologies knowledge. The retrieved knowledge and conversation context are used to provide live coaching feedback on Tone, Empathy, and Clarity, as well as a pre-generated draft reply. The human agent retains complete control to review, edit, and approve the response, while SQLite maintains full persistent audit logs."*

---

## ❓ 2. Core Viva Questions & Model Answers

### Q1: What is the primary role of AI in this system? Does it replace the human agent?
**Answer:**  
No, our system implements a **Human-in-the-Loop** architecture. The AI acts as an intelligent co-pilot for human customer support agents. In high-stakes domains (like e-commerce and consumer tech), fully autonomous chatbots can frustrate customers, make false promises, or hallucinate policies. Our system empowers the human agent with real-time analytics, grounded policy search, draft generation, and draft quality scoring, while the human agent remains the final authority on every sent message.

---

### Q2: How does your system support English, Hindi, and Telugu?
**Answer:**  
We employ a hybrid multilingual architecture:
1. **Script-Based Detection**: Native Telugu (`\u0C00-\u0C7F`) and Devanagari/Hindi (`\u0900-\u097F`) characters have distinct Unicode ranges. Inspecting character code points provides $O(N)$ instantaneous, 100% accurate script classification without loading heavy language models.
2. **Statistical N-Gram Detection**: For Romanized or mixed text (e.g. Hindi written in English letters), `langdetect` acts as a statistical fallback.
3. **Cross-Lingual RAG Mapping**: Queries in Indic languages are mapped to canonical English knowledge concepts, allowing single-corpus knowledge retrieval.
4. **Multilingual Response Generation**: The AI formulates grounded draft suggestions in the customer's detected language (English, Hindi, or Telugu).

---

### Q3: What is Zero-Shot Intent Classification and how is it used here?
**Answer:**  
In traditional supervised classification, an AI model requires thousands of pre-labeled examples per category and cannot recognize new classes without retraining. In **Zero-Shot Intent Classification**, candidate categories (e.g. `Refund Request`, `Technical Issue`, `Cancellation`) are represented as semantic text descriptions. The model computes the semantic similarity between the customer's query and each intent's semantic embedding space, categorizing the message without needing model retraining.

---

### Q4: What is RAG (Retrieval-Augmented Generation) and why use FAISS instead of standard SQL `LIKE` search?
**Answer:**  
- **RAG** combines dense information retrieval with language generation to ground AI responses in verified company documents, preventing policy fabrication.
- **Why FAISS over SQL `LIKE`?** Keyword searching (e.g. `WHERE text LIKE '%broken%'`) fails when users use synonyms (e.g., *"router red light"* vs *"connectivity failure"* vs *"wan modem blink"*). **FAISS (Facebook AI Similarity Search)** embeds text into continuous vector spaces where semantically similar phrases cluster together, enabling conceptual search in sub-millisecond time.

---

### Q5: How is Cosine Similarity computed in FAISS using `IndexFlatIP`?
**Answer:**  
`IndexFlatIP` computes the exact **Inner Product** (dot product) between vectors:
$$\text{IP}(u, v) = \sum_{i=1}^d u_i v_i$$
Before adding embeddings to FAISS, our embedding service normalizes each vector using $L_2$ normalization:
$$\|u\|_2 = \sqrt{\sum u_i^2} = 1.0$$
Because both vectors have unit length, the Inner Product is mathematically identical to **Cosine Similarity**:
$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v = \text{IP}(u, v)$$

---

### Q6: How does the system guarantee Zero Policy Hallucination?
**Answer:**  
We enforce a strict **Relevance Threshold** ($\ge 0.35$ cosine similarity). If a customer asks a question outside the Nexora Technologies knowledge base (e.g. *"What is the recipe for cookies?"*), the maximum similarity score drops below the threshold. Instead of guessing, the retriever explicitly triggers a **No-Answer Fallback**: *"No relevant company information was found in the Nexora Technologies knowledge base. Please verify this information manually."*

---

### Q7: Explain the Two-Way AI Coaching Engine.
**Answer:**  
Our coaching engine works in two distinct modes:
1. **Customer-Context Guidance**: Analyzes the customer's sentiment, urgency, escalation risk, and language to provide upfront coaching tips (e.g., *"Customer is frustrated — prioritize empathy and avoid defensive language"*).
2. **Agent-Draft Real-Time Evaluation**: As the agent types their response in the composer, the system continuously analyzes the draft and calculates three real-time scores ($1\text{-}10$):
   - **Tone Score**: Measures politeness and checks for forbidden defensive phrases (*"not our fault"*, *"read the manual"*).
   - **Empathy Score**: Checks if the agent acknowledged frustration and apologized sincerely.
   - **Clarity Score**: Rewards specific business timelines (*"within 24-48 business hours"*) and penalizes vague words (*"soon"*, *"later"*).

---

### Q8: How does the system validate and inspect uploaded images?
**Answer:**  
1. **Validation**: Verifies file existence, checks against an extension whitelist (`.jpg`, `.jpeg`, `.png`), enforces a $5\text{MB}$ size ceiling, and runs `PIL.Image.verify()` to catch corrupted file streams.
2. **Visual Diagnostics**: Analyzes image dimensions, format, and optical color distributions (such as dominant red pixel clusters on router hardware photos) to diagnose hardware LED status and checkout error screens.

---

### Q9: Why decouple Streamlit `st.session_state` from SQLite?
**Answer:**  
- **`st.session_state`** is stored in temporary RAM and tied only to the active browser tab session. It allows instantaneous UI re-rendering, active draft typing, and reactive metric updates.
- **SQLite Database** (`support_history.db`) is permanent disk storage. It saves complete multi-turn conversation logs, quality scores, and retrieved sources for supervisor audits and performance reporting even after the browser tab is closed.

---

### Q10: What are the future scope enhancements for this project?
**Answer:**  
1. **Voice Integration**: Adding Speech-to-Text (Whisper) for real-time customer voice call transcription.
2. **Additional Indian Languages**: Extending support to Tamil, Kannada, Marathi, and Bengali.
3. **CRM Integration**: Connecting via webhooks to CRM platforms like Salesforce Service Cloud or Zendesk.
4. **Supervisor Analytics Dashboard**: Aggregated weekly reports on agent empathy trends and common escalation triggers.
