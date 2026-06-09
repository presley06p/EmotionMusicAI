"""
Spotify Web API integration.

Set the following environment variables (or a .env file):
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
    SPOTIFY_REDIRECT_URI   (default: http://localhost:5000/spotify/callback)
"""

import os
import base64
import logging
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

CLIENT_ID     = os.environ.get("c29d8561a03f48ddb074a2c6ce2d4d3e", "")
CLIENT_SECRET = os.environ.get("88d23060019c4798a974d2e367660731", "")
REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:5000/spotify/callback")

SCOPES = " ".join([
    "user-read-private",
    "user-read-email",
    "streaming",
    "user-modify-playback-state",
    "playlist-read-private",
])

AUTH_URL  = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE  = "https://api.spotify.com/v1"


# ── Emotion → search query mapping ───────────────────────────────────────────
EMOTION_QUERIES = {
    "happy":    ["happy upbeat pop hits", "feel good dance", "summer vibes playlist"],
    "sad":      ["sad acoustic songs", "emotional ballads", "rainy day music"],
    "angry":    ["rock anthems", "metal playlist", "intense workout music"],
    "fear":     ["calm instrumental", "meditation music", "peaceful ambient"],
    "love":     ["romantic songs", "love ballads", "date night playlist"],
    "surprise": ["unexpected hits", "genre-bending music", "eclectic mix"],
    "neutral":  ["top charts", "trending songs", "popular hits 2024"],
    "disgust":  ["rock anthems", "metal playlist", "intense workout music"],
}

EMOTION_SEED_GENRES = {
    "happy":    ["pop", "dance", "happy"],
    "sad":      ["acoustic", "sad", "piano"],
    "angry":    ["rock", "metal", "punk"],
    "fear":     ["ambient", "classical", "sleep"],
    "love":     ["romance", "soul", "r-n-b"],
    "surprise": ["indie", "alternative", "jazz"],
    "neutral":  ["pop", "hip-hop", "indie"],
}


# ── Auth helpers ──────────────────────────────────────────────────────────────
def get_spotify_auth_url() -> str:
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def get_spotify_token(code: str) -> dict | None:
    if not CLIENT_ID or not CLIENT_SECRET:
        logger.warning("[Spotify] CLIENT_ID or CLIENT_SECRET not set.")
        return None

    credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type":  "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type":   "authorization_code",
        "code":         code,
        "redirect_uri": REDIRECT_URI,
    }

    try:
        resp = requests.post(TOKEN_URL, headers=headers, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[Spotify] Token exchange error: {e}")
        return None


def _client_credentials_token() -> str | None:
    """Get an app-level token (no user auth needed) for search queries."""
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type":  "application/x-www-form-urlencoded",
    }
    try:
        resp = requests.post(
            TOKEN_URL, headers=headers,
            data={"grant_type": "client_credentials"}, timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        logger.error(f"[Spotify] Client credentials error: {e}")
        return None


# ── Track helpers ─────────────────────────────────────────────────────────────
def _format_track(item: dict) -> dict:
    album_images = item.get("album", {}).get("images", [])
    album_cover  = album_images[0]["url"] if album_images else ""
    duration_ms  = item.get("duration_ms", 0)
    minutes, seconds = divmod(duration_ms // 1000, 60)

    return {
        "id":          item["id"],
        "name":        item["name"],
        "artist":      ", ".join(a["name"] for a in item.get("artists", [])),
        "album":       item.get("album", {}).get("name", ""),
        "album_cover": album_cover,
        "duration":    f"{minutes}:{seconds:02d}",
        "spotify_url": item.get("external_urls", {}).get("spotify", ""),
        "preview_url": item.get("preview_url") or "",
        "uri":         item.get("uri", ""),
    }


def search_tracks_by_emotion(emotion: str, limit: int = 12) -> list:
    """Search Spotify for tracks matching the given emotion."""
    queries = EMOTION_QUERIES.get(emotion, EMOTION_QUERIES["neutral"])
    token   = _client_credentials_token()

    if not token:
        logger.warning("[Spotify] No token; returning mock tracks.")
        return _mock_tracks(emotion, limit)

    tracks  = []
    headers = {"Authorization": f"Bearer {token}"}

    for query in queries[:2]:           # 2 queries × 6 results
        try:
            resp = requests.get(
                f"{API_BASE}/search",
                headers=headers,
                params={"q": query, "type": "track", "limit": limit // 2},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("tracks", {}).get("items", [])
            tracks.extend([_format_track(t) for t in items])
        except Exception as e:
            logger.error(f"[Spotify] Search error for '{query}': {e}")

    # Deduplicate by id, return up to `limit`
    seen = set()
    result = []
    for t in tracks:
        if t["id"] not in seen:
            seen.add(t["id"])
            result.append(t)
        if len(result) >= limit:
            break

    return result or _mock_tracks(emotion, limit)


def get_recommendations(emotion: str, seed_tracks: list = None, limit: int = 10) -> list:
    """Use Spotify's recommendations endpoint seeded by genre + mood."""
    token = _client_credentials_token()
    if not token:
        return _mock_tracks(emotion, limit)

    genres  = EMOTION_SEED_GENRES.get(emotion, ["pop"])[:2]
    headers = {"Authorization": f"Bearer {token}"}
    params  = {
        "seed_genres": ",".join(genres),
        "limit":       limit,
    }

    # Valence / energy tuning by emotion
    mood_params = {
        "happy":    {"min_valence": 0.7, "min_energy": 0.6},
        "sad":      {"max_valence": 0.4, "max_energy": 0.5},
        "angry":    {"min_energy": 0.8, "max_valence": 0.5},
        "fear":     {"max_energy": 0.4, "target_instrumentalness": 0.6},
        "love":     {"min_valence": 0.6, "max_energy": 0.7},
        "surprise": {},
        "neutral":  {"target_valence": 0.5},
    }
    params.update(mood_params.get(emotion, {}))

    try:
        resp = requests.get(
            f"{API_BASE}/recommendations",
            headers=headers, params=params, timeout=10
        )
        resp.raise_for_status()
        tracks = resp.json().get("tracks", [])
        return [_format_track(t) for t in tracks]
    except Exception as e:
        logger.error(f"[Spotify] Recommendations error: {e}")
        return _mock_tracks(emotion, limit)


# ── Mock data for demo / no-credentials environments ─────────────────────────
MOCK_LIBRARY = {
    "happy": [
        ("Happy", "Pharrell Williams", "G I R L"),
        ("Can't Stop the Feeling!", "Justin Timberlake", "Trolls"),
        ("Shake It Off", "Taylor Swift", "1989"),
        ("Uptown Funk", "Mark Ronson ft. Bruno Mars", "Uptown Special"),
        ("Good as Hell", "Lizzo", "Cuz I Love You"),
        ("Levitating", "Dua Lipa", "Future Nostalgia"),
    ],
    "sad": [
        ("Someone Like You", "Adele", "21"),
        ("Fix You", "Coldplay", "X&Y"),
        ("Skinny Love", "Bon Iver", "For Emma, Forever Ago"),
        ("The Night Will Always Win", "Manchester Orchestra", "Hope"),
        ("Falling", "Harry Styles", "Fine Line"),
        ("Liability", "Lorde", "Melodrama"),
    ],
    "angry": [
        ("Break Stuff", "Limp Bizkit", "Significant Other"),
        ("Killing in the Name", "Rage Against the Machine", "Rage Against the Machine"),
        ("Given Up", "Linkin Park", "Minutes to Midnight"),
        ("Chop Suey!", "System of a Down", "Toxicity"),
        ("Figured You Out", "Nickelback", "The Long Road"),
        ("Bulls on Parade", "Rage Against the Machine", "Evil Empire"),
    ],
    "fear": [
        ("Weightless", "Marconi Union", "Weightless"),
        ("Clair de Lune", "Debussy", "Suite bergamasque"),
        ("River Flows in You", "Yiruma", "First Love"),
        ("Experience", "Ludovico Einaudi", "In a Time Lapse"),
        ("Gymnopédie No.1", "Erik Satie", "Gymnopédies"),
        ("Spiegel im Spiegel", "Arvo Pärt", "Tabula Rasa"),
    ],
    "love": [
        ("Perfect", "Ed Sheeran", "÷"),
        ("At Last", "Etta James", "At Last!"),
        ("Can't Help Falling in Love", "Elvis Presley", "Blue Hawaii"),
        ("All of Me", "John Legend", "Love in the Future"),
        ("Make You Feel My Love", "Adele", "19"),
        ("Thinking Out Loud", "Ed Sheeran", "x"),
    ],
    "surprise": [
        ("Take Five", "Dave Brubeck Quartet", "Time Out"),
        ("Bohemian Rhapsody", "Queen", "A Night at the Opera"),
        ("Mr. Brightside", "The Killers", "Hot Fuss"),
        ("Stairway to Heaven", "Led Zeppelin", "Led Zeppelin IV"),
        ("Purple Rain", "Prince", "Purple Rain"),
        ("Heroes", "David Bowie", "'Heroes'"),
    ],
    "neutral": [
        ("Blinding Lights", "The Weeknd", "After Hours"),
        ("As It Was", "Harry Styles", "Harry's House"),
        ("Anti-Hero", "Taylor Swift", "Midnights"),
        ("Flowers", "Miley Cyrus", "Endless Summer Vacation"),
        ("Kill Bill", "SZA", "SOS"),
        ("Escapism.", "RAYE ft. 070 Shake", "My 21st Century Blues"),
    ],
}

COVERS = [
    "https://via.placeholder.com/300/1DB954/FFFFFF?text=🎵",
    "https://via.placeholder.com/300/191414/1DB954?text=♪",
    "https://via.placeholder.com/300/535353/FFFFFF?text=🎶",
]

def _mock_tracks(emotion: str, limit: int = 6) -> list:
    library = MOCK_LIBRARY.get(emotion, MOCK_LIBRARY["neutral"])
    tracks  = []
    for i, (name, artist, album) in enumerate(library[:limit]):
        tracks.append({
            "id":          f"mock_{emotion}_{i}",
            "name":        name,
            "artist":      artist,
            "album":       album,
            "album_cover": COVERS[i % len(COVERS)],
            "duration":    f"{3 + (i % 2)}:{(i * 17) % 60:02d}",
            "spotify_url": "https://open.spotify.com",
            "preview_url": "",
            "uri":         "",
            "mock":        True,
        })
    return tracks
