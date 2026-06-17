"""
Text Emotion Analysis — powered by Groq API (LLaMA 3).
Falls back to enhanced keyword model if API key not set.
"""

import re
import os
import json
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

EMOTIONS = ["happy", "sad", "angry", "fear", "love", "surprise", "neutral"]

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": ["happy", "joy", "excited", "amazing", "awesome", "great", "smile", "laugh"],
    "sad": ["sad", "cry", "lonely", "depressed", "hurt", "pain", "tears"],
    "angry": ["angry", "rage", "furious", "mad", "hate", "frustrated"],
    "fear": ["scared", "afraid", "anxious", "worried", "panic", "nervous"],
    "love": ["love", "romantic", "crush", "affection", "heart"],
    "surprise": ["surprised", "shocked", "amazed", "wow", "unexpected"],
    "neutral": ["okay", "fine", "normal", "average", "meh"],
}

NEGATIONS = {
    "not", "no", "never", "don't", "can't", "won't", "isn't", "wasn't"
}

INTENSIFIERS = {
    "very": 1.5,
    "really": 1.5,
    "extremely": 2.0,
    "so": 1.3,
    "super": 1.4,
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _keyword_scores(tokens: List[str]) -> Dict[str, float]:
    scores = {e: 0.0 for e in EMOTION_KEYWORDS}

    negated = False
    neg_window = 0
    intensity = 1.0

    for tok in tokens:

        if tok in NEGATIONS:
            negated = True
            neg_window = 3
            continue

        if tok in INTENSIFIERS:
            intensity = INTENSIFIERS[tok]
            continue

        for emotion, keywords in EMOTION_KEYWORDS.items():
            if tok in keywords:
                delta = intensity * (-0.6 if negated else 1.0)
                scores[emotion] += delta

        if negated:
            neg_window -= 1
            if neg_window <= 0:
                negated = False

        intensity = 1.0

    return scores


def _analyze_with_groq(text: str) -> dict | None:
    if not GROQ_API_KEY:
        return None

    prompt = f"""
Analyze the emotion in the text.

Return ONLY valid JSON:
{{
  "emotion": "happy",
  "confidence": 0.9,
  "explanation": "short reason",
  "all_scores": {{
    "happy": 0.9,
    "sad": 0.0,
    "angry": 0.0,
    "fear": 0.0,
    "love": 0.1,
    "surprise": 0.0,
    "neutral": 0.0
  }}
}}

Text: "{text}"
"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 400,
            },
            timeout=15,
        )

        if resp.status_code != 200:
            logger.error(f"Groq error: {resp.text}")
            return None

        raw = resp.json()["choices"][0]["message"]["content"].strip()

        # clean markdown
        raw = re.sub(r"```json|```", "", raw).strip()

        data = json.loads(raw)

        emotion = data.get("emotion", "neutral")
        if emotion not in EMOTIONS:
            emotion = "neutral"

        scores = data.get("all_scores", {})

        for e in EMOTIONS:
            scores.setdefault(e, 0.0)

        total = sum(scores.values()) or 1

        scores = {k: round(v / total, 4) for k, v in scores.items()}

        return {
            "emotion": emotion,
            "confidence": round(float(data.get("confidence", 0.7)), 4),
            "all_scores": scores,
            "explanation": data.get("explanation", ""),
            "method": "groq-ai",
        }

    except Exception as e:
        logger.error(f"Groq exception: {e}")
        return None


def analyze_text_emotion(text: str) -> dict:
    print("Using Groq model...")

    result = _analyze_with_groq(text)
    if result:
        return result

    print("Falling back to keyword model...")

    tokens = _tokenize(text)
    scores = _keyword_scores(tokens)

    total = sum(max(v, 0) for v in scores.values()) or 1
    norm = {k: round(max(v, 0) / total, 4) for k, v in scores.items()}

    best_emotion = max(norm, key=norm.get)
    confidence = norm[best_emotion]

    if confidence == 0:
        best_emotion = "neutral"
        confidence = 0.5
        norm["neutral"] = 0.5

    EXPLANATIONS = {
        "happy": "Your message radiates positivity and joy. 🎉",
        "sad": "Your message expresses sadness or loneliness. 💙",
        "angry": "Your message conveys frustration or anger. 🔥",
        "fear": "Your message suggests anxiety or worry. 😰",
        "love": "Your message is filled with warmth and affection. ❤️",
        "surprise": "Your message reflects astonishment. 😮",
        "neutral": "Your message has a calm, neutral tone. 😐",
    }

    return {
        "emotion": best_emotion,
        "confidence": confidence,
        "all_scores": norm,
        "explanation": EXPLANATIONS.get(best_emotion, ""),
        "method": "keyword",
    }