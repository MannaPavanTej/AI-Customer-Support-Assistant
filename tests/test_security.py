"""Tests for local production-security primitives."""

import unittest
from cryptography.fernet import Fernet
from services.security_service import DataCipher, hash_password, verify_password


class TestSecurityService(unittest.TestCase):
    def test_password_hash_verification(self):
        encoded = hash_password("correct-horse-battery-staple")
        self.assertNotIn("correct-horse", encoded)
        self.assertTrue(verify_password("correct-horse-battery-staple", encoded))
        self.assertFalse(verify_password("incorrect", encoded))

    def test_encryption_round_trip(self):
        cipher = DataCipher(Fernet.generate_key().decode())
        encrypted = cipher.encrypt("customer@example.com")
        self.assertTrue(encrypted.startswith("enc:"))
        self.assertEqual(cipher.decrypt(encrypted), "customer@example.com")


if __name__ == "__main__":
    unittest.main()
