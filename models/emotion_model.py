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

EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
]
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": ["happy", "joy", "excited", "amazing", "awesome", "great", "smile", "laugh"],
    "sad": ["sad", "cry", "lonely", "depressed", "hurt", "pain", "tears"],
    "angry": ["angry", "rage", "furious", "mad", "hate", "frustrated"],
    "fear": ["scared", "afraid", "anxious", "worried", "panic", "nervous"],
    "love": ["love", "romantic", "crush", "affection", "heart"],
    "surprise": ["surprised", "shocked", "amazed", "wow", "unexpected"],
    "neutral": ["okay", "fine", "normal", "average", "meh"],
}

NEGATIONS = {"not", "no", "never", "don't", "can't", "won't", "isn't", "wasn't"}

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
Analyze the emotion.

Return ONLY valid JSON:

{{
  "emotion":"joy",
  "confidence":0.95,
  "explanation":"...",
  "all_scores": {{
      "admiration":0,
      "amusement":0,
      "anger":0,
      "annoyance":0,
      "approval":0,
      "caring":0,
      "confusion":0,
      "curiosity":0,
      "desire":0,
      "disappointment":0,
      "disapproval":0,
      "disgust":0,
      "embarrassment":0,
      "excitement":0,
      "fear":0,
      "gratitude":0,
      "grief":0,
      "joy":0,
      "love":0,
      "nervousness":0,
      "optimism":0,
      "pride":0,
      "realization":0,
      "relief":0,
      "remorse":0,
      "sadness":0,
      "surprise":0,
      "neutral":0
  }}
}}

Text:
"{text}"
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

        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(raw)
        except Exception:
            logger.error(f"JSON parsing failed: {raw}")
            return None

            emotion = data.get("emotion", "neutral")
        if emotion not in EMOTIONS:
            emotion = "neutral"

        scores = data.get("all_scores", {})

        clean_scores = {}

        for emo in EMOTIONS:
            try:
                clean_scores[emo] = float(scores.get(emo, 0))
            except:
                clean_scores[emo] = 0.0

        total = sum(clean_scores.values())

        if total <= 0:
            clean_scores["neutral"] = 1.0
            total = 1.0

        scores = {
            emo: round(value / total, 4)
            for emo, value in clean_scores.items()
        }

        confidence = max(
            round(float(data.get("confidence", 0.7)), 4),
            max(scores.values())
        )

        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_scores": scores,
            "explanation": data.get(
                "explanation",
                f"The detected emotion is {emotion}."
            ),
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