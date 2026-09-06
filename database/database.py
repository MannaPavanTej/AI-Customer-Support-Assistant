"""
SQLite Database Connection and Schema Management.
"""

import sqlite3
from pathlib import Path
from typing import Optional
import config


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.

    Args:
        db_path: Optional custom path (useful for test databases)

    Returns:
        sqlite3.Connection object with row_factory set to sqlite3.Row
    """
    path = db_path or config.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initializes the database schema if tables do not exist.

    Args:
        db_path: Optional custom path for testing
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        primary_language TEXT DEFAULT 'en',
        status TEXT DEFAULT 'OPEN',
        total_messages INTEGER DEFAULT 0
    );
    """)

    # 2. Conversation Logs Table (Audit and Metrics Record)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        customer_message TEXT,
        agent_message TEXT,
        language TEXT,
        sentiment TEXT,
        sentiment_confidence REAL,
        intent TEXT,
        intent_confidence REAL,
        urgency TEXT,
        escalation_risk TEXT,
        key_issue TEXT,
        retrieved_sources TEXT,
        coaching_tip TEXT,
        tone_score INTEGER,
        empathy_score INTEGER,
        clarity_score INTEGER,
        suggested_response TEXT,
        image_path TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
    );
    """)

    # Create Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON conversation_logs (session_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON conversation_logs (timestamp);")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin', 'agent', 'viewer')),
        active INTEGER NOT NULL DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        username TEXT,
        action TEXT NOT NULL,
        details TEXT
    );
    """)

    conn.commit()
    conn.close()
