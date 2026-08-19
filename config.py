"""
Central configuration settings for the AI-Powered Customer Support Assistant.
Contains project constants, file paths, supported languages, and default parameters.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DATABASE_DIR = BASE_DIR / "database"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

# Ensure runtime directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Application Info
APP_TITLE = "TechCare Live Support Co-Pilot"
APP_SUBTITLE = "AI-Powered Live Guidance for Support Agents"
COMPANY_NAME = "TechCare Inc."

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "te": "Telugu (తెలుగు)"
}
DEFAULT_LANGUAGE = "en"

# Database Configuration
DATABASE_PATH = DATABASE_DIR / "support_history.db"

# Thresholds & Limits
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
MAX_HISTORY_DISPLAY = 20
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_IMAGE_SIZE_MB = 5.0
