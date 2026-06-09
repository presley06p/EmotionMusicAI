"""
Text Emotion Analysis using a keyword + transformer-lite approach.

Falls back to a rule-based lexicon so the app works even without GPU / heavy
ML libraries installed.  When `transformers` is available it uses a
zero-shot or sentiment pipeline for richer classification.
"""

import re
from typing import Dict, List


# ── Keyword Lexicon (fast fallback) ──────────────────────────────────────────
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": [
        "happy", "joy", "joyful", "excited", "elated", "cheerful", "delighted",
        "wonderful", "great", "fantastic", "amazing", "awesome", "love", "thrilled",
        "ecstatic", "blissful", "euphoric", "grateful", "blessed", "positive",
        "glad", "pleased", "content", "satisfied", "celebrate", "smile", "laugh"
    ],
    "sad": [
        "sad", "unhappy", "depressed", "down", "blue", "gloomy", "miserable",
        "lonely", "hopeless", "heartbroken", "cry", "crying", "tears", "grief",
        "sorrow", "melancholy", "dejected", "despair", "hurt", "pain", "lost",
        "alone", "empty", "broken", "miss", "missing", "regret"
    ],
    "angry": [
        "angry", "anger", "furious", "rage", "mad", "irritated", "frustrated",
        "annoyed", "hatred", "hate", "disgusted", "livid", "enraged", "hostile",
        "bitter", "resentful", "stressed", "stress", "tension", "overwhelmed",
        "infuriated", "outraged", "violent", "aggressive"
    ],
    "fear": [
        "scared", "fear", "afraid", "anxious", "anxiety", "nervous", "worried",
        "panic", "terrified", "dread", "horror", "phobia", "uneasy", "tense",
        "apprehensive", "insecure", "vulnerable", "helpless", "trembling", "shaking"
    ],
    "love": [
        "love", "romantic", "romance", "crush", "affection", "adore", "cherish",
        "passionate", "devotion", "tender", "sweet", "darling", "intimate",
        "together", "heart", "caring", "compassion", "warmth", "relationship"
    ],
    "surprise": [
        "surprised", "surprise", "shocked", "astonished", "amazed", "stunned",
        "unexpected", "sudden", "wow", "unbelievable", "incredible", "disbelief",
        "astounded", "speechless", "whoa", "omg", "what"
    ],
    "neutral": [
        "okay", "fine", "alright", "normal", "usual", "regular", "average",
        "nothing", "just", "day", "today", "went", "going"
    ],
}

# Negation words that flip the dominant emotion score
NEGATIONS = {"not", "no", "never", "neither", "nor", "nobody", "nothing",
             "nowhere", "neither", "hardly", "barely", "scarcely"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _keyword_scores(tokens: List[str]) -> Dict[str, float]:
    scores = {e: 0.0 for e in EMOTION_KEYWORDS}
    negated = False

    for i, tok in enumerate(tokens):
        if tok in NEGATIONS:
            negated = True
            continue

        for emotion, keywords in EMOTION_KEYWORDS.items():
            if tok in keywords:
                delta = -0.5 if negated else 1.0
                scores[emotion] += delta

        # Negation window: reset after 3 tokens
        if negated and i > 0:
            negated = False

    return scores


def _build_explanation(emotion: str, confidence: float, text: str) -> str:
    templates = {
        "happy":   "Your message radiates positivity and joy. 🎉",
        "sad":     "Your message expresses sadness or loneliness. 💙",
        "angry":   "Your message conveys frustration or anger. 🔥",
        "fear":    "Your message suggests anxiety or worry. 😰",
        "love":    "Your message is filled with warmth and affection. ❤️",
        "surprise":"Your message reflects astonishment or unexpected news. 😮",
        "neutral": "Your message has a calm, neutral tone. 😐",
    }
    base = templates.get(emotion, "Emotion detected from your text.")
    conf_str = f"Confidence: {confidence:.0%}."
    return f"{base} {conf_str}"


# ── Optional transformer pipeline ────────────────────────────────────────────
_pipeline = None

def _try_load_transformer():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        print("[NLP] Transformer pipeline loaded.")
    except Exception:
        _pipeline = False  # mark as unavailable
    return _pipeline


# ── Public API ────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "joy":     "happy",
    "sadness": "sad",
    "anger":   "angry",
    "fear":    "fear",
    "love":    "love",
    "surprise":"surprise",
    "neutral": "neutral",
    "disgust": "angry",
}


def analyze_text_emotion(text: str) -> dict:
    """
    Classify the emotion in *text*.

    Returns:
        {
          "emotion":     str,
          "confidence":  float (0-1),
          "all_scores":  {emotion: score, ...},
          "explanation": str,
        }
    """
    # Try transformer first
    pipe = _try_load_transformer()
    if pipe and pipe is not False:
        try:
            results = pipe(text[:512])[0]  # list of {label, score}
            all_scores = {
                LABEL_MAP.get(r["label"].lower(), r["label"].lower()): r["score"]
                for r in results
            }
            best = max(results, key=lambda r: r["score"])
            emotion    = LABEL_MAP.get(best["label"].lower(), best["label"].lower())
            confidence = best["score"]
            return {
                "emotion":     emotion,
                "confidence":  round(confidence, 4),
                "all_scores":  {k: round(v, 4) for k, v in all_scores.items()},
                "explanation": _build_explanation(emotion, confidence, text),
                "method":      "transformer",
            }
        except Exception as exc:
            print(f"[NLP] Transformer error: {exc}; falling back to keyword.")

    # Keyword fallback
    tokens = _tokenize(text)
    scores = _keyword_scores(tokens)

    total = sum(max(v, 0) for v in scores.values()) or 1
    norm  = {k: round(max(v, 0) / total, 4) for k, v in scores.items()}

    best_emotion = max(norm, key=norm.get)
    confidence   = norm[best_emotion]

    # If no keyword matched, default to neutral
    if confidence == 0:
        best_emotion = "neutral"
        confidence   = 0.5
        norm["neutral"] = 0.5

    return {
        "emotion":     best_emotion,
        "confidence":  confidence,
        "all_scores":  norm,
        "explanation": _build_explanation(best_emotion, confidence, text),
        "method":      "keyword",
    }
