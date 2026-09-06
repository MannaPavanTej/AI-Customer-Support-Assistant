"""Local authentication, encryption, and safe password helpers."""

import base64
import hashlib
import hmac
import os
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
import config


def hash_password(password: str) -> str:
    """Return a salted PBKDF2-SHA256 password hash; never store plaintext."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return "pbkdf2_sha256$310000$%s$%s" % (
        base64.urlsafe_b64encode(salt).decode(), base64.urlsafe_b64encode(digest).decode()
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.urlsafe_b64decode(salt), int(rounds))
        return hmac.compare_digest(base64.urlsafe_b64encode(digest).decode(), expected)
    except (ValueError, TypeError):
        return False


class DataCipher:
    """Encrypts audit content when DATA_ENCRYPTION_KEY is configured."""
    def __init__(self, key: Optional[str] = None):
        self._fernet = Fernet((key or config.DATA_ENCRYPTION_KEY).encode()) if (key or config.DATA_ENCRYPTION_KEY) else None

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt(self, value: Optional[str]) -> Optional[str]:
        if value is None or not self._fernet:
            return value
        return "enc:" + self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: Optional[str]) -> Optional[str]:
        if value is None or not value.startswith("enc:"):
            return value
        if not self._fernet:
            return "[Encrypted content unavailable: configure DATA_ENCRYPTION_KEY]"
        try:
            return self._fernet.decrypt(value[4:].encode()).decode()
        except InvalidToken:
            return "[Encrypted content could not be decrypted]"
