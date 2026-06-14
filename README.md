# ◈ EmoTune AI — Emotion-Based Music Recommendation System

> Detect your emotion via webcam or text, get a perfect Spotify playlist instantly.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Spotify](https://img.shields.io/badge/Spotify-API-1DB954)
![AI](https://img.shields.io/badge/AI-FER%20%2B%20NLP-orange)

---

## Features

| Feature | Details |
|---|---|
| 😊 Facial Emotion | Webcam → FER + MTCNN → 7 emotions with confidence |
| 💬 Text Emotion | NLP keyword / transformer classification |
| 🎵 Spotify Integration | Dynamic track search by emotion |
| 📊 Dashboard | Charts, history, saved songs |
| 🌓 Dark/Light Mode | Persisted via localStorage |
| 🔐 Auth | Register, login, session, password hashing |

---

## Project Structure

```
EmotionMusicAI/
├── app.py                    # Flask app + all routes
├── requirements.txt
├── gunicorn.conf.py
├── Procfile
├── .env.example
│
├── database/
│   └── db.py                 # SQLite init + helpers
│
├── models/
│   ├── emotion_model.py      # Text emotion analysis (NLP)
│   └── face_detection.py     # Facial emotion (FER/DeepFace)
│
├── spotify/
│   └── spotify_api.py        # Spotify auth + search
│
├── static/
│   ├── css/
│   │   ├── main.css          # Full theme + components
│   │   └── detect.css        # Detection page styles
│   └── js/
│       ├── main.js           # Theme toggle, nav
│       └── detect.js         # Webcam, text analysis, music
│
└── templates/
    ├── base.html             # Shared layout + navbar + footer
    ├── index.html            # Landing page
    ├── login.html
    ├── register.html
    ├── emotion_detection.html
    └── dashboard.html
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yourname/EmotionMusicAI.git
cd EmotionMusicAI

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Spotify credentials and a strong SECRET_KEY
```

### 3. Set Up Spotify API

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Set **Redirect URI** → `http://localhost:5000/spotify/callback`
4. Copy **Client ID** and **Client Secret** into your `.env`

> Without Spotify credentials the app still works — it falls back to
> curated demo tracks so you can see the full UI immediately.

### 4. Run

```bash
python app.py
# Open http://localhost:5000
```

---

## AI / ML Setup

### Facial Emotion (FER)

The app uses the `fer` library with MTCNN face detection:

```bash
pip install fer tensorflow opencv-python-headless
```

If FER is unavailable (e.g. no GPU), it gracefully falls back to a **demo mode** that returns plausible random results for testing.

**Optional — DeepFace (more accurate):**

```bash
pip install deepface
```

### Text Emotion

**Default:** Fast keyword-lexicon classifier (zero dependencies, works immediately).

**Upgrade to transformer model** (much more accurate, ~800 MB download):

```bash
pip install transformers torch
```

The app auto-detects which is available.

---

## Emotion → Music Mapping

| Emotion  | Genres / Vibe               |
|----------|-----------------------------|
| Happy    | Pop, Dance, Upbeat          |
| Sad      | Acoustic, Ballads, Piano    |
| Angry    | Rock, Metal, Punk           |
| Fear     | Ambient, Classical, Calm    |
| Love     | Romance, Soul, R&B          |
| Surprise | Indie, Alternative, Jazz    |
| Neutral  | Top Charts, Trending        |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Landing page |
| GET/POST | `/register` | User registration |
| GET/POST | `/login` | User login |
| GET | `/logout` | Logout |
| GET | `/detect` | Emotion detection page |
| GET | `/dashboard` | User dashboard |
| POST | `/api/analyze-text` | Text emotion + Spotify tracks |
| POST | `/api/analyze-face` | Image emotion + Spotify tracks |
| POST | `/api/save-song` | Save song to favourites |
| GET | `/api/history` | Fetch detection history (JSON) |
| DELETE | `/api/history/<id>` | Delete history entry |
| GET | `/spotify/login` | Start Spotify OAuth flow |
| GET | `/spotify/callback` | Spotify OAuth callback |

---

## Database Schema

```sql
-- Users
CREATE TABLE users (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  username  TEXT NOT NULL UNIQUE,
  email     TEXT NOT NULL UNIQUE,
  password  TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Emotion detection history
CREATE TABLE emotion_history (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id    INTEGER NOT NULL,
  emotion    TEXT NOT NULL,
  confidence REAL NOT NULL,
  source     TEXT NOT NULL,  -- 'text' or 'face'
  timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Saved song recommendations
CREATE TABLE recommendations (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id     INTEGER NOT NULL,
  song_name   TEXT NOT NULL,
  artist      TEXT NOT NULL,
  emotion     TEXT NOT NULL,
  spotify_url TEXT,
  album_cover TEXT,
  timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Deployment

### Render

1. Push to GitHub
2. New Web Service → connect repo
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `gunicorn app:app -c gunicorn.conf.py`
5. Add environment variables from `.env`

### Railway

1. `railway init && railway up`
2. Add env vars in the Railway dashboard

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
```

---

## Security Notes

- Passwords hashed with `werkzeug.security.generate_password_hash` (PBKDF2-SHA256)
- Flask sessions with `SECRET_KEY` — set a strong random key in production
- All protected routes use `@login_required` decorator
- User input validated before DB writes
- No plaintext credentials stored anywhere

---

## Extending the App

- **Voice input:** Use the Web Speech API in `detect.js` to transcribe speech → feed to `/api/analyze-text`
- **Admin dashboard:** Add an `is_admin` column to `users`, protect `/admin` routes
- **Real Spotify playback:** After OAuth, use `session['spotify_token']` with the Spotify Web Playback SDK
- **Push notifications:** Use Flask-SocketIO for real-time mood alerts

---

## License

MIT — feel free to use, modify, and deploy.
