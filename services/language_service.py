"""
Language Detection Service supporting English, Hindi, and Telugu.

Uses a hybrid approach:
1. Fast Unicode Script Analysis (instantly and accurately classifies native Telugu & Hindi text)
2. Statistical N-Gram Language Detection (langdetect) for Romanized or mixed text
"""

import re
from typing import Tuple
from langdetect import detect_langs, DetectorFactory
import config

# Enforce deterministic results from langdetect
DetectorFactory.seed = 42

# Unicode Ranges
TELUGU_RANGE = re.compile(r'[\u0C00-\u0C7F]')
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> Tuple[str, str, float]:
    """
    Detects whether the input text is in English, Hindi, or Telugu.

    Args:
        text: Customer message string

    Returns:
        Tuple containing:
        - language_code: 'en', 'hi', or 'te'
        - language_name: 'English', 'Hindi', or 'Telugu'
        - confidence: Confidence score (0.0 to 1.0)
    """
    if not text or not text.strip():
        return config.DEFAULT_LANGUAGE, config.SUPPORTED_LANGUAGES[config.DEFAULT_LANGUAGE], 1.0

    cleaned = text.strip()

    # 1. Fast Script-Based Detection for Native Indic Scripts
    telugu_chars = len(TELUGU_RANGE.findall(cleaned))
    devanagari_chars = len(DEVANAGARI_RANGE.findall(cleaned))
    total_alpha = len(re.findall(r'\w', cleaned)) or 1

    if telugu_chars / total_alpha > 0.2:
        return "te", config.SUPPORTED_LANGUAGES.get("te", "Telugu"), round(min(0.99, (telugu_chars / total_alpha) + 0.1), 2)

    if devanagari_chars / total_alpha > 0.2:
        return "hi", config.SUPPORTED_LANGUAGES.get("hi", "Hindi"), round(min(0.99, (devanagari_chars / total_alpha) + 0.1), 2)

    # 2. Statistical Fallback via langdetect
    try:
        predictions = detect_langs(cleaned)
        if predictions:
            top_pred = predictions[0]
            code = top_pred.lang
            confidence = round(top_pred.prob, 2)

            if code in config.SUPPORTED_LANGUAGES:
                return code, config.SUPPORTED_LANGUAGES[code], confidence
            elif code in ["mr", "ne"]:  # Devanagari close relatives fallback to Hindi
                return "hi", config.SUPPORTED_LANGUAGES["hi"], confidence
    except Exception:
        pass

    # Default fallback to English
    return config.DEFAULT_LANGUAGE, config.SUPPORTED_LANGUAGES[config.DEFAULT_LANGUAGE], 0.85
