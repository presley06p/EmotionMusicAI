"""
Emotion-Based Music Recommendation System
Main Flask Application Entry Point
"""

# Standard Python imports
import os
import json
import base64
import logging
import tempfile
import subprocess
from pathlib import Path

# Environment variables
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
from datetime import datetime, timedelta
from functools import wraps

# Third-party imports
import imageio_ffmpeg
from flask import request, jsonify

# Project imports
from models.voice_emotion import detect_voice_emotion

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response
)
from werkzeug.security import generate_password_hash, check_password_hash

from database.db import init_db, get_db
from models.emotion_model import analyze_text_emotion
from models.face_detection import analyze_face_emotion
from spotify.spotify_api import (
    get_spotify_auth_url, get_spotify_token,
    search_tracks_by_emotion, get_recommendations,
    has_credentials
)

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emotion-music-ai-secret-2024")
app.config["SESSION_PERMANENT"] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Auth Decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Routes: Public ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", user=_current_user())


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        db = get_db()
        existing = db.execute(
            "SELECT id FROM users WHERE email=? OR username=?", (email, username)
        ).fetchone()

        if existing:
            flash("Username or email already exists.", "danger")
            return render_template("register.html")

        pw_hash = generate_password_hash(password)
        db.execute(
            "INSERT INTO users (username, email, password) VALUES (?,?,?)",
            (username, email, pw_hash)
        )
        db.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password   = request.form.get("password", "")

        db   = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email=? OR username=?",
            (identifier, identifier)
        ).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid credentials.", "danger")

    return render_template("login.html")





@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "info")
    return redirect(url_for("index"))


# ── Routes: Protected ─────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    db   = get_db()
    uid  = session["user_id"]

    history = db.execute(
        "SELECT * FROM emotion_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
        (uid,)
    ).fetchall()

    recs = db.execute(
        "SELECT * FROM recommendations WHERE user_id=? ORDER BY timestamp DESC LIMIT 10",
        (uid,)
    ).fetchall()

    # Emotion frequency for charts
    emotion_freq = db.execute(
        """SELECT emotion, COUNT(*) as count FROM emotion_history
           WHERE user_id=? GROUP BY emotion ORDER BY count DESC""",
        (uid,)
    ).fetchall()

    # Weekly mood data (last 7 days)
    weekly = []
    for i in range(6, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        row = db.execute(
            """SELECT emotion, COUNT(*) as count FROM emotion_history
               WHERE user_id=? AND date(timestamp)=? GROUP BY emotion ORDER BY count DESC LIMIT 1""",
            (uid, day)
        ).fetchone()
        weekly.append({
            "date": day,
            "emotion": row["emotion"] if row else "None",
            "count": row["count"] if row else 0
        })

    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

    return render_template(
        "dashboard.html",
        user=user,
        history=history,
        recommendations=recs,
        emotion_freq=emotion_freq,
        weekly=weekly
    )

# Load Spotify credentials
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

print("CLIENT_ID loaded:", bool(CLIENT_ID))
print("CLIENT_SECRET loaded:", bool(CLIENT_SECRET))

@app.route("/detect")
@login_required
def detect():
    ai_key = bool(os.environ.get("GROQ_API_KEY"))
    spotify_ok = has_credentials()
    return render_template("emotion_detection.html", user=_current_user(), ai_enabled=ai_key, spotify_enabled=spotify_ok)



# ── API: Text Emotion ─────────────────────────────────────────────────────────
@app.route("/api/analyze-text", methods=["POST"])
@login_required
def api_analyze_text():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = analyze_text_emotion(text)

    # Persist
    db = get_db()
    db.execute(
        "INSERT INTO emotion_history (user_id, emotion, confidence, source, timestamp) VALUES (?,?,?,?,?)",
        (session["user_id"], result["emotion"], result["confidence"], "text", datetime.utcnow())
    )
    db.commit()

    # Fetch songs
    tracks = search_tracks_by_emotion(result["emotion"])
    result["tracks"] = tracks

    return jsonify(result)


# ── API: Face Emotion ─────────────────────────────────────────────────────────
@app.route("/api/analyze-face", methods=["POST"])
@login_required
def api_analyze_face():
    data     = request.get_json()
    img_data = (data or {}).get("image", "")

    if not img_data:
        return jsonify({"error": "No image data"}), 400

    # Strip data-URL prefix
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    result = analyze_face_emotion(img_data)

    if result.get("emotion") and result["emotion"] != "no_face":
            # Persist
     db = get_db()
    db.execute(
        "INSERT INTO emotion_history (user_id, emotion, confidence, source, timestamp) VALUES (?,?,?,?,?)",
        (
            session["user_id"],
            result["emotion"],
            result["confidence"],
            "text",
            datetime.utcnow()
        )
    )
    db.commit()

    print("History saved successfully")

    row = db.execute(
        "SELECT COUNT(*) FROM emotion_history WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()

    print("Total history:", row[0])

    # Fetch songs
    tracks = search_tracks_by_emotion(result["emotion"])
    result["tracks"] = tracks

    return jsonify(result)


# ── API: Save Favourite ───────────────────────────────────────────────────────
@app.route("/api/save-song", methods=["POST"])
@login_required
def api_save_song():
    data = request.get_json() or {}
    required = ("song_name", "artist", "emotion", "spotify_url")
    if not all(data.get(k) for k in required):
        return jsonify({"error": "Missing fields"}), 400

    db = get_db()
    db.execute(
        """INSERT INTO recommendations
           (user_id, song_name, artist, emotion, spotify_url, album_cover, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (
            session["user_id"],
            data["song_name"], data["artist"],
            data["emotion"],   data["spotify_url"],
            data.get("album_cover", ""),
            datetime.utcnow()
        )
    )
    db.commit()
    return jsonify({"success": True})


# ── API: History ──────────────────────────────────────────────────────────────
@app.route("/api/history")
@login_required
def api_history():
    db   = get_db()
    uid  = session["user_id"]

    history = db.execute(
        "SELECT * FROM emotion_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 50",
        (uid,)
    ).fetchall()

    return jsonify([dict(row) for row in history])


# ── API: Delete History Entry ─────────────────────────────────────────────────
@app.route("/api/history/<int:entry_id>", methods=["DELETE"])
@login_required
def api_delete_history(entry_id):
    db = get_db()
    db.execute(
        "DELETE FROM emotion_history WHERE id=? AND user_id=?",
        (entry_id, session["user_id"])
    )
    db.commit()
    return jsonify({"success": True})


# ── Spotify OAuth ─────────────────────────────────────────────────────────────
@app.route("/spotify/login")
@login_required
def spotify_login():
    auth_url = get_spotify_auth_url()
    return redirect(auth_url)


@app.route("/spotify/callback")
@login_required
def spotify_callback():
    code  = request.args.get("code")
    error = request.args.get("error")

    if error or not code:
        flash("Spotify authorisation failed.", "danger")
        return redirect(url_for("dashboard"))

    token_info = get_spotify_token(code)
    if token_info:
        session["spotify_token"]   = token_info["access_token"]
        session["spotify_refresh"] = token_info.get("refresh_token")
        flash("Spotify connected!", "success")
    else:
        flash("Could not get Spotify token.", "danger")

    return redirect(url_for("detect"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _current_user():
    if "user_id" not in session:
        return None
    return {"id": session["user_id"], "username": session.get("username", "")}


# ── Init & Run ────────────────────────────────────────────────────────────────
# Initialize database when app starts
with app.app_context():
    init_db()


@app.route('/analyze-voice', methods=['POST'])
def analyze_voice():
    webm_path = None
    wav_path = None

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files['audio']

    try:
        # Save browser recording as WebM
        with tempfile.NamedTemporaryFile(
            suffix='.webm', delete=False
        ) as temp:
            audio_file.save(temp.name)
            webm_path = temp.name

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as temp:
            wav_path = temp.name

        # Convert WebM to WAV using FFmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", webm_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                wav_path
            ],
            check=True,
            capture_output=True,
            text=True
        )

        # Detect emotion
        result = detect_voice_emotion(wav_path)

        # Fetch Spotify songs based on detected voice emotion
        tracks = search_tracks_by_emotion(result["emotion"])

        # Save voice emotion to history
        db = get_db()
        db.execute(
            """INSERT INTO emotion_history
               (user_id, emotion, confidence, source, timestamp)
               VALUES (?,?,?,?,?)""",
            (
                session["user_id"],
                result["emotion"],
                result["confidence"],
                "voice",
                datetime.utcnow()
            )
        )
        db.commit()

        return jsonify({
            "success": True,
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "method": "wav2vec2",
            "tracks": tracks
        })

    except subprocess.CalledProcessError as e:
        app.logger.error("FFmpeg error: %s", e.stderr)
        return jsonify({
            "success": False,
            "error": "Audio conversion failed."
        }), 400

    except Exception as e:
        app.logger.exception("Voice analysis failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        for path in [webm_path, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

