
import librosa
import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification
)

MODEL_ID = "Dpngtm/wav2vec2-emotion-recognition"

# Load model only once
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

model.eval()


def detect_voice_emotion(audio_path):

    # Load audio, convert to mono and resample to 16 kHz
    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Reject recordings that are too short or silent
    if len(audio) < 16000:
        raise ValueError("Please record at least one second of speech.")

    if np.max(np.abs(audio)) < 0.005:
        raise ValueError("No clear speech detected.")

    # Model recommends speech recordings under 10 seconds
    audio = audio[:16000 * 10]

    # Extract audio input for Wav2Vec2
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Predict emotion
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )[0]

    predicted_id = torch.argmax(probabilities).item()

    emotion = model.config.id2label[predicted_id]

    confidence = probabilities[predicted_id].item()

    return {
        "emotion": emotion.lower(),
        "confidence": round(confidence * 100, 2)
    }
