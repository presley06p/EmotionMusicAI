"""
Facial Emotion Detection — powered by Google Gemini Vision (free).
Falls back to varied demo mode if no API key.
"""

import base64
import json
import logging
import os
import re
import random
import requests

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

EMOTIONS = ["happy", "sad", "angry", "fear", "love", "surprise", "neutral"]


def _analyze_with_gemini_vision(b64_image: str) -> dict | None:
    """Send webcam frame to Gemini Vision for emotion analysis."""
    if not GEMINI_API_KEY:
        return None

    prompt = """You are an expert facial emotion analyst. Analyse the face in this image carefully.

Look at: eyebrows (raised/furrowed), mouth (smile/frown/open), eyes (wide/squinted/tearful), forehead tension, cheek position.

Return ONLY this JSON with no other text or markdown:
{
  "emotion": "one of: happy, sad, angry, fear, love, surprise, neutral",
  "confidence": 0.0,
  "face_detected": true,
  "explanation": "one sentence describing what facial features indicate this emotion",
  "all_scores": {
    "happy": 0.0,
    "sad": 0.0,
    "angry": 0.0,
    "fear": 0.0,
    "love": 0.0,
    "surprise": 0.0,
    "neutral": 0.0
  }
}

Rules:
- all_scores must sum to 1.0
- If no face visible, set face_detected to false
- Do NOT default to happy or neutral — be precise"""

    try:
        resp = requests.post(
            f"{GEMINI_VISION_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 400,
                }
            },
            timeout=15,
        )
        resp.raise_for_status()

        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data = json.loads(raw)

        if not data.get("face_detected", True):
            return {"emotion": "no_face", "confidence": 0,
                    "all_scores": {}, "face_box": None}

        emotion = data.get("emotion", "neutral")
        if emotion not in EMOTIONS:
            emotion = "neutral"

        all_scores = data.get("all_scores", {})
        for e in EMOTIONS:
            if e not in all_scores:
                all_scores[e] = 0.0

        total = sum(all_scores.values()) or 1
        all_scores = {k: round(v / total, 4) for k, v in all_scores.items()}

        return {
            "emotion":     emotion,
            "confidence":  round(float(data.get("confidence", 0.7)), 4),
            "all_scores":  all_scores,
            "face_box":    [50, 50, 200, 200],
            "explanation": data.get("explanation", ""),
            "method":      "gemini-vision",
        }

    except json.JSONDecodeError as e:
        logger.error(f"[FaceDetect] JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"[FaceDetect] Gemini Vision error: {e}")
        return None


def _enhanced_fallback() -> dict:
    """Varied demo result — not biased to happy/neutral."""
    weights = {
        "happy": 0.22, "neutral": 0.18, "sad": 0.16,
        "angry": 0.14, "fear": 0.12, "surprise": 0.10, "love": 0.08,
    }
    noisy = {k: max(0.01, v + random.gauss(0, 0.05)) for k, v in weights.items()}
    total = sum(noisy.values())
    scores = {k: round(v / total, 4) for k, v in noisy.items()}
    dominant = max(scores, key=scores.get)

    return {
        "emotion":    dominant,
        "confidence": round(scores[dominant], 4),
        "all_scores": scores,
        "face_box":   [50, 50, 200, 200],
        "mock":       True,
        "method":     "demo",
    }


def analyze_face_emotion(b64_image: str) -> dict:
    """Analyse webcam frame. Uses Gemini Vision if key set, else demo."""
    result = _analyze_with_gemini_vision(b64_image)
    if result:
        return result

    logger.warning("[FaceDetect] No GEMINI_API_KEY — using demo mode.")
    return _enhanced_fallback()