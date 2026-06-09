"""
Facial Emotion Detection module.

Uses the `fer` library (which wraps TensorFlow/Keras + OpenCV) when available.
Falls back to DeepFace, and finally returns a mock result for demo purposes.
"""

import base64
import io
import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing heavy deps ──────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed; face detection degraded.")

try:
    FER_AVAILABLE = False
_fer_detector = None

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("[DeepFace] DeepFace loaded.")
except Exception:
    DEEPFACE_AVAILABLE = False


# Canonical emotion labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Maps FER / DeepFace labels → canonical names
FER_MAP = {
    "angry":    "angry",
    "disgust":  "angry",    # map disgust → angry for music matching
    "fear":     "fear",
    "happy":    "happy",
    "sad":      "sad",
    "surprise": "surprise",
    "neutral":  "neutral",
}


def _decode_image(b64_data: str) -> Optional[np.ndarray]:
    """Decode a base64 image string into an OpenCV BGR numpy array."""
    try:
        img_bytes = base64.b64decode(b64_data)
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        if CV2_AVAILABLE:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return img
        else:
            # Pillow fallback
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(img)[:, :, ::-1]  # RGB → BGR
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return None


def _fer_analyze(img: np.ndarray) -> Optional[dict]:
    """Analyse frame with the FER library."""
    try:
        result = _fer_detector.detect_emotions(img)
        if not result:
            return None

        # Use the face with the largest bounding box
        best = max(result, key=lambda r: r["box"][2] * r["box"][3])
        emotions = best["emotions"]  # {"happy": 0.97, "sad": 0.01, ...}

        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        normalized = {FER_MAP.get(k, k): round(v, 4) for k, v in emotions.items()}

        return {
            "emotion":     FER_MAP.get(dominant, dominant),
            "confidence":  round(confidence, 4),
            "all_scores":  normalized,
            "face_box":    best["box"],
        }
    except Exception as e:
        logger.error(f"FER analysis error: {e}")
        return None


def _deepface_analyze(img: np.ndarray) -> Optional[dict]:
    """Analyse frame with DeepFace as fallback."""
    try:
        result = DeepFace.analyze(
            img, actions=["emotion"], enforce_detection=False, silent=True
        )
        if isinstance(result, list):
            result = result[0]

        emotions = result["emotion"]  # {"angry": 12.3, "happy": 87.4, ...}
        dominant = result["dominant_emotion"]

        total = sum(emotions.values()) or 1
        normalized = {
            FER_MAP.get(k.lower(), k.lower()): round(v / total, 4)
            for k, v in emotions.items()
        }
        confidence = normalized.get(FER_MAP.get(dominant.lower(), dominant.lower()), 0.5)

        return {
            "emotion":    FER_MAP.get(dominant.lower(), dominant.lower()),
            "confidence": round(confidence, 4),
            "all_scores": normalized,
            "face_box":   None,
        }
    except Exception as e:
        logger.error(f"DeepFace error: {e}")
        return None


def _mock_result() -> dict:
    """Return a plausible random result for demo / test environments."""
    weights = [0.12, 0.05, 0.10, 0.35, 0.15, 0.10, 0.13]
    probs   = [random.gauss(w, 0.03) for w in weights]
    probs   = [max(0, p) for p in probs]
    total   = sum(probs) or 1
    probs   = [p / total for p in probs]

    scores = dict(zip(EMOTIONS, probs))
    dominant = max(scores, key=scores.get)

    return {
        "emotion":     dominant,
        "confidence":  round(scores[dominant], 4),
        "all_scores":  {k: round(v, 4) for k, v in scores.items()},
        "face_box":    [50, 50, 200, 200],
        "mock":        True,
    }


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_face_emotion(b64_image: str) -> dict:
    """
    Analyse a base64-encoded image and return emotion data.

    Returns dict with keys:
        emotion, confidence, all_scores, face_box, (optional) mock
    On failure or no face detected: {"emotion": "no_face", ...}
    """
    img = _decode_image(b64_image)

    if img is None:
        return {"emotion": "no_face", "confidence": 0, "all_scores": {}, "error": "decode_failed"}

    # Attempt FER
    if FER_AVAILABLE and _fer_detector:
        result = _fer_analyze(img)
        if result:
            return result

    # Attempt DeepFace
    if DEEPFACE_AVAILABLE:
        result = _deepface_analyze(img)
        if result:
            return result

    # Demo fallback — always returns something useful
    logger.warning("[FaceDetect] No ML backend available; using mock result.")
    return _mock_result()
