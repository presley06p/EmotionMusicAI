"""
Facial Emotion Detection — Demo mode.
Returns realistic random results when no ML backend is available.
Full FER/DeepFace support can be added locally with TensorFlow.
"""

import random
import logging

logger = logging.getLogger(__name__)

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_WEIGHTS = {
    "happy":    0.30,
    "neutral":  0.20,
    "sad":      0.18,
    "angry":    0.12,
    "fear":     0.10,
    "surprise": 0.10,
}

def analyze_face_emotion(b64_image: str) -> dict:
    """
    Returns a demo emotion result.
    Replace this with FER/DeepFace locally if TensorFlow is available.
    """
    emotions = list(EMOTION_WEIGHTS.keys())
    weights  = list(EMOTION_WEIGHTS.values())

    # Generate realistic-looking scores
    raw = [random.gauss(w, 0.04) for w in weights]
    raw = [max(0.01, r) for r in raw]
    total = sum(raw)
    scores = {e: round(r / total, 4) for e, r in zip(emotions, raw)}

    dominant   = max(scores, key=scores.get)
    confidence = scores[dominant]

    logger.info(f"[FaceDetect] Demo mode — detected: {dominant} ({confidence:.0%})")

    return {
        "emotion":    dominant,
        "confidence": round(confidence, 4),
        "all_scores": scores,
        "face_box":   [60, 40, 180, 180],
        "mock":       True,
    }
