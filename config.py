"""
Central configuration settings for the AI-Powered Customer Support Assistant.
Contains project constants, file paths, supported languages, and default parameters.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DATABASE_DIR = BASE_DIR / "database"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
LOG_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Application Info
APP_TITLE = "Development of AI-Powered Customer Support Assistant with Live Response Guidance"
APP_SUBTITLE = "Human-guided intelligence for exceptional customer support"
COMPANY_NAME = "Nexora Technologies"

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
MAX_UPLOAD_AGE_HOURS = 24
EMBEDDING_DIMENSION = 384

# Security switches. Development remains frictionless; production requires an
# explicit administrator account and a Fernet key for encrypted audit content.
APP_ENV = os.getenv("APP_ENV", "development").lower()
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")
DATA_ENCRYPTION_KEY = os.getenv("DATA_ENCRYPTION_KEY", "")


def validate_production_configuration() -> None:
    if APP_ENV == "production":
        missing = []
        if not AUTH_ENABLED:
            missing.append("AUTH_ENABLED=true")
        if not ADMIN_USERNAME or not ADMIN_PASSWORD:
            missing.append("ADMIN_USERNAME and ADMIN_PASSWORD")
        if not DATA_ENCRYPTION_KEY:
            missing.append("DATA_ENCRYPTION_KEY")
        if missing:
            raise RuntimeError("Production configuration is incomplete: " + ", ".join(missing))
