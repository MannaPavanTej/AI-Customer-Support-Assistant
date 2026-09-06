"""
Support Database Repository.
Provides CRUD data access operations for conversation sessions, message turns, and audit logs.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid
from datetime import datetime
from models.conversation import ConversationState
from .database import get_connection, init_db
from services.security_service import DataCipher, hash_password, verify_password
from services.observability import get_logger


class SupportRepository:
    """
    Data access repository for Nexora Technologies conversation persistence.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        init_db(self.db_path)
        self.cipher = DataCipher()
        self.logger = get_logger()

    def bootstrap_admin(self, username: str, password: str, role: str = "admin") -> None:
        """Create the configured administrator once; password is stored hashed."""
        if not username or not password:
            return
        conn = get_connection(self.db_path)
        conn.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                     (username, hash_password(password), role))
        conn.commit()
        conn.close()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, str]]:
        conn = get_connection(self.db_path)
        row = conn.execute("SELECT username, password_hash, role FROM users WHERE username = ? AND active = 1", (username,)).fetchone()
        conn.close()
        if row and verify_password(password, row["password_hash"]):
            self.log_audit(username, "login_success")
            return {"username": row["username"], "role": row["role"]}
        self.log_audit(username or "unknown", "login_failed")
        return None

    def log_audit(self, username: str, action: str, details: str = "") -> None:
        conn = get_connection(self.db_path)
        conn.execute("INSERT INTO audit_events (username, action, details) VALUES (?, ?, ?)", (username, action, details[:500]))
        conn.commit()
        conn.close()
        self.logger.info("audit action=%s username=%s", action, username)

    def create_session(self, session_id: Optional[str] = None, language: str = "en") -> str:
        """
        Creates a new support conversation session in the database.

        Args:
            session_id: Optional existing ID, otherwise generates a UUID
            language: Initial primary language code

        Returns:
            The session_id string
        """
        sid = session_id or f"TC-{uuid.uuid4().hex[:8].upper()}"
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id, primary_language, status)
        VALUES (?, ?, 'OPEN');
        """, (sid, language))

        conn.commit()
        conn.close()
        return sid

    def log_turn(
        self,
        session_id: str,
        state: ConversationState,
        agent_message: str,
        image_path: Optional[str] = None
    ) -> int:
        """
        Persists a complete conversation turn and its AI analysis metrics.

        Args:
            session_id: Unique session identifier
            state: ConversationState instance containing analysis metrics
            agent_message: The final approved message sent by the human agent
            image_path: Optional customer uploaded image path

        Returns:
            log_id of the inserted row
        """
        last_cust_msg = state.last_customer_message
        customer_text = last_cust_msg.text if last_cust_msg else ""
        img = image_path or (last_cust_msg.image_path if last_cust_msg else None)

        sources_json = json.dumps(state.retrieved_sources)
        coaching_tip = state.coaching.coaching_tip if state.coaching else ""
        tone = state.coaching.tone_score if state.coaching else 8
        empathy = state.coaching.empathy_score if state.coaching else 8
        clarity = state.coaching.clarity_score if state.coaching else 8

        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        # Ensure session exists
        cursor.execute("""
        INSERT OR IGNORE INTO sessions (session_id, primary_language, status)
        VALUES (?, ?, 'OPEN');
        """, (session_id, state.language))

        # Insert turn record
        cursor.execute("""
        INSERT INTO conversation_logs (
            session_id, customer_message, agent_message, language,
            sentiment, sentiment_confidence, intent, intent_confidence,
            urgency, escalation_risk, key_issue, retrieved_sources,
            coaching_tip, tone_score, empathy_score, clarity_score,
            suggested_response, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            session_id, self.cipher.encrypt(customer_text), self.cipher.encrypt(agent_message), state.language,
            state.sentiment, state.sentiment_confidence, state.intent, state.intent_confidence,
            state.urgency, state.escalation_risk, self.cipher.encrypt(state.key_issue), sources_json,
            coaching_tip, tone, empathy, clarity,
            self.cipher.encrypt(state.suggested_response), self.cipher.encrypt(img)
        ))

        log_id = cursor.lastrowid

        # Update session timestamp and total count
        cursor.execute("""
        UPDATE sessions
        SET updated_at = CURRENT_TIMESTAMP,
            primary_language = ?,
            total_messages = total_messages + 2
        WHERE session_id = ?;
        """, (state.language, session_id))

        conn.commit()
        conn.close()
        self.logger.info("turn_logged session_id=%s encrypted=%s", session_id, self.cipher.enabled)
        return log_id

    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves a list of recent conversation sessions.

        Returns:
            List of session summary dicts
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT session_id, created_at, updated_at, primary_language, status, total_messages
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT ?;
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all detailed log entries for a given session.

        Args:
            session_id: Target session ID

        Returns:
            List of log records
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT * FROM conversation_logs
        WHERE session_id = ?
        ORDER BY log_id ASC;
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            d = dict(r)
            try:
                d["retrieved_sources"] = json.loads(d["retrieved_sources"]) if d["retrieved_sources"] else []
            except Exception:
                d["retrieved_sources"] = []
            for field in ("customer_message", "agent_message", "key_issue", "suggested_response", "image_path"):
                d[field] = self.cipher.decrypt(d.get(field))
            results.append(d)

        return results

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session and its associated logs.

        Args:
            session_id: Target session ID

        Returns:
            True if deleted successfully
        """
        conn = get_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM conversation_logs WHERE session_id = ?;", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))

        conn.commit()
        conn.close()
        return True

    def export_session_markdown(self, session_id: str) -> str:
        """
        Generates a clean Markdown transcript of a past session.

        Args:
            session_id: Target session ID

        Returns:
            Formatted Markdown transcript string
        """
        logs = self.get_session_logs(session_id)
        if not logs:
            return f"# Nexora Technologies Support Session: {session_id}\n\n*No conversation records found.*"

        lines = [
            f"# Nexora Technologies Support Audit Transcript",
            f"**Session ID:** `{session_id}`  ",
            f"**Total Turns:** {len(logs)}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---",
            "## Conversation Turns & Metrics\n"
        ]

        for idx, log in enumerate(logs, 1):
            lines.append(f"### Turn {idx} `[{log['timestamp']}]`")
            lines.append(f"- **Language:** {log['language'].upper()} | **Sentiment:** {log['sentiment']} ({int(log['sentiment_confidence']*100)}%)")
            lines.append(f"- **Intent:** {log['intent']} | **Urgency:** {log['urgency']} | **Risk:** {log['escalation_risk']}")
            lines.append(f"- **Key Issue:** {log['key_issue']}")
            lines.append(f"\n> **👤 Customer:** {log['customer_message']}")
            lines.append(f"> **🎧 Agent (Final Approved):** {log['agent_message']}\n")
            lines.append(f"**Coaching Evaluation:** Tone: {log['tone_score']}/10 | Empathy: {log['empathy_score']}/10 | Clarity: {log['clarity_score']}/10")
            lines.append(f"*Tip:* {log['coaching_tip']}")
            lines.append("\n---\n")

        return "\n".join(lines)
