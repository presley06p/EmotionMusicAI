"""
Text Emotion Analysis — powered by Google Gemini API (free).
Falls back to enhanced keyword model if API key not set.
"""

import re
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
import json
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)


EMOTIONS = ["happy", "sad", "angry", "fear", "love", "surprise", "neutral"]

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": [
        "happy", "joy", "joyful", "excited", "elated", "cheerful", "delighted",
        "wonderful", "great", "fantastic", "amazing", "awesome", "thrilled",
        "ecstatic", "blissful", "euphoric", "grateful", "blessed", "positive",
        "glad", "pleased", "content", "satisfied", "celebrate", "smile", "laugh",
        "overjoyed", "gleeful", "jubilant", "radiant", "energetic", "pumped",
    ],
    "sad": [
        "sad", "unhappy", "depressed", "down", "blue", "gloomy", "miserable",
        "lonely", "hopeless", "heartbroken", "cry", "crying", "tears", "grief",
        "sorrow", "melancholy", "dejected", "despair", "hurt", "pain", "lost",
        "alone", "empty", "broken", "miss", "missing", "regret", "mourn",
        "devastated", "crushed", "bitter", "aching", "numb", "hollow",
        "worthless", "meaningless", "tired", "exhausted", "drained",
    ],
    "angry": [
        "angry", "anger", "furious", "rage", "mad", "irritated", "frustrated",
        "annoyed", "hatred", "hate", "disgusted", "livid", "enraged", "hostile",
        "bitter", "resentful", "stressed", "stress", "tension", "overwhelmed",
        "infuriated", "outraged", "aggressive", "seething", "boiling", "fuming",
        "irate", "fed up", "unfair", "betrayed", "cheated", "offended",
    ],
    "fear": [
        "scared", "fear", "afraid", "anxious", "anxiety", "nervous", "worried",
        "panic", "terrified", "dread", "horror", "uneasy", "tense",
        "apprehensive", "insecure", "vulnerable", "helpless", "trembling",
        "shaking", "paranoid", "threatened", "unsafe", "uncertain", "dreading",
        "paralysed", "frozen", "exam", "test", "deadline", "failing", "petrified",
    ],
    "love": [
        "love", "romantic", "romance", "crush", "affection", "adore", "cherish",
        "passionate", "devotion", "tender", "sweet", "darling", "intimate",
        "together", "heart", "caring", "compassion", "warmth", "relationship",
        "infatuated", "smitten", "butterflies", "miss you", "attracted",
        "enchanted", "captivated", "falling for", "soul mate", "beloved",
    ],
    "surprise": [
        "surprised", "surprise", "shocked", "astonished", "amazed", "stunned",
        "unexpected", "sudden", "wow", "unbelievable", "incredible", "disbelief",
        "astounded", "speechless", "whoa", "omg", "no way", "seriously",
        "can't believe", "blown away", "mind blown", "out of nowhere",
        "jaw dropped", "floored", "dumbfounded",
    ],
    "neutral": [
        "okay", "fine", "alright", "normal", "usual", "regular", "average",
        "nothing special", "so so", "meh", "indifferent", "whatever",
        "routine", "ordinary", "bland", "uneventful", "standard",
    ],
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "wasn't", "weren't",
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.5, "extremely": 2.0, "incredibly": 2.0,
    "so": 1.3, "absolutely": 1.8, "totally": 1.4, "completely": 1.6,
    "deeply": 1.5, "super": 1.4, "utterly": 1.7,
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _keyword_scores(tokens: List[str]) -> Dict[str, float]:
    scores = {e: 0.0 for e in EMOTION_KEYWORDS}
    negated = False
    negation_window = 0
    intensity = 1.0

    for tok in tokens:
        if tok in NEGATIONS:
            negated = True
            negation_window = 4
            continue
        if tok in INTENSIFIERS:
            intensity = INTENSIFIERS[tok]
            continue

        matched = False
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if tok in keywords:
                delta = intensity * (-0.6 if negated else 1.0)
                scores[emotion] += delta
                matched = True

        if matched:
            intensity = 1.0

        if negated:
            negation_window -= 1
            if negation_window <= 0:
                negated = False

    return scores


def _analyze_with_groq(text: str) -> dict | None:
    print("Groq key loaded:", bool(GROQ_API_KEY))
    if not GROQ_API_KEY:
        return None

    prompt = f"""Analyse the emotion in this text deeply.
Consider sarcasm, implied meaning, context, and nuance.

Text: "{text}"

Return ONLY this JSON, no other text:
{{
  "emotion": "one of: happy, sad, angry, fear, love, surprise, neutral",
  "confidence": 0.85,
  "explanation": "2 sentences explaining the emotion and any nuance",
  "all_scores": {{
    "happy": 0.0,
    "sad": 0.0,
    "angry": 0.0,
    "fear": 0.0,
    "love": 0.0,
    "surprise": 0.0,
    "neutral": 0.0
  }}
}}
all_scores must sum to 1.0"""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 400,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data = json.loads(raw)
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
            "explanation": data.get("explanation", ""),
            "method":      "groq-ai",
        }

    except json.JSONDecodeError as e:
        logger.error(f"[Groq] JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"[Groq] API error: {e}")
        return None


def analyze_text_emotion(text: str) -> dict:
    result = _analyze_with_groq(text)
    if result:
        return result

    # Keyword fallback
    tokens = _tokenize(text)
    scores = _keyword_scores(tokens)
    total = sum(max(v, 0) for v in scores.values()) or 1
    norm  = {k: round(max(v, 0) / total, 4) for k, v in scores.items()}
    best_emotion = max(norm, key=norm.get)
    confidence   = norm[best_emotion]
    if confidence == 0:
        best_emotion = "neutral"
        confidence   = 0.5
        norm["neutral"] = 0.5

    EXPLANATIONS = {
        "happy":    "Your message radiates positivity and joy. 🎉",
        "sad":      "Your message expresses sadness or loneliness. 💙",
        "angry":    "Your message conveys frustration or anger. 🔥",
        "fear":     "Your message suggests anxiety or worry. 😰",
        "love":     "Your message is filled with warmth and affection. ❤️",
        "surprise": "Your message reflects astonishment. 😮",
        "neutral":  "Your message has a calm, neutral tone. 😐",
    }
    return {
        "emotion":     best_emotion,
        "confidence":  confidence,
        "all_scores":  norm,
        "explanation": EXPLANATIONS.get(best_emotion, ""),
        "method":      "keyword",
    }