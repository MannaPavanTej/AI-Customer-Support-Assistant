"""
Database package for persistent SQLite storage.
"""

from .database import get_connection, init_db
from .repository import SupportRepository

__all__ = ["get_connection", "init_db", "SupportRepository"]
