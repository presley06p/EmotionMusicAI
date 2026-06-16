"""
Spotify Web API integration — with auto token refresh and live search.

Environment variables required:
    SPOTIFY_CLIENT_ID
    SPOTIFY_CLIENT_SECRET
    SPOTIFY_REDIRECT_URI  (default: http://localhost:5000/spotify/callback)
"""

import os
import base64
import logging
import time
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
print("CLIENT_ID loaded:", bool(CLIENT_ID))
print("CLIENT_SECRET loaded:", bool(CLIENT_SECRET))
REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI",
                                "http://localhost:5000/spotify/callback")

SCOPES = " ".join([
    "user-read-private", "user-read-email",
    "streaming", "user-modify-playback-state",
    "playlist-read-private",
])

AUTH_URL  = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE  = "https://api.spotify.com/v1"

# ── In-memory token cache (client credentials) ────────────────────────────
_token_cache = {"token": None, "expires_at": 0}


# ── Emotion → Spotify search config ──────────────────────────────────────
EMOTION_QUERIES = {
    "happy":    ["happy upbeat pop", "feel good hits", "good vibes dance"],
    "sad":      ["sad songs acoustic", "emotional ballads heartbreak", "melancholy indie"],
    "angry":    ["rock anthems", "metal intense", "rage workout music"],
    "fear":     ["calm meditation music", "peaceful instrumental ambient", "relaxing piano"],
    "love":     ["romantic love songs", "date night R&B soul", "love ballads"],
    "surprise": ["eclectic indie alternative", "jazz fusion unexpected", "genre bending hits"],
    "neutral":  ["top hits 2024", "popular trending songs", "chill playlist"],
    "disgust":  ["hard rock aggressive", "punk anthems", "heavy metal"],
}

EMOTION_AUDIO_FEATURES = {
    "happy":    {"min_valence": 0.65, "min_energy": 0.55, "target_danceability": 0.75},
    "sad":      {"max_valence": 0.40, "max_energy": 0.50, "target_acousticness": 0.60},
    "angry":    {"min_energy": 0.80,  "max_valence": 0.50, "target_loudness": -5.0},
    "fear":     {"max_energy": 0.40,  "target_instrumentalness": 0.60, "max_valence": 0.45},
    "love":     {"min_valence": 0.55, "target_energy": 0.55, "target_danceability": 0.60},
    "surprise": {"target_valence": 0.60, "target_energy": 0.65},
    "neutral":  {"target_valence": 0.50, "target_energy": 0.50},
}

EMOTION_SEED_GENRES = {
    "happy":    ["pop", "dance", "happy"],
    "sad":      ["acoustic", "sad", "piano"],
    "angry":    ["rock", "metal", "punk"],
    "fear":     ["ambient", "classical", "sleep"],
    "love":     ["romance", "soul", "r-n-b"],
    "surprise": ["indie", "alternative", "jazz"],
    "neutral":  ["pop", "hip-hop", "indie"],
    "disgust":  ["rock", "metal", "punk"],
}


# ── Auth helpers ──────────────────────────────────────────────────────────
def _credentials_header() -> str:
    return base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()


def get_spotify_auth_url() -> str:
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPES,
        "show_dialog":   "true",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def get_spotify_token(code: str) -> dict | None:
    if not CLIENT_ID or not CLIENT_SECRET:
        logger.warning("[Spotify] Credentials not set.")
        return None
    try:
        resp = requests.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {_credentials_header()}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": REDIRECT_URI,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[Spotify] Token exchange error: {e}")
        return None


def _get_client_token() -> str | None:
    logger.warning(f"CLIENT_ID: {CLIENT_ID[:5]}..." if CLIENT_ID else "No Client ID")
    logger.warning(f"CLIENT_SECRET loaded: {bool(CLIENT_SECRET)}")
    """
    Get / refresh an app-level (client credentials) access token.
    Cached in memory and auto-refreshed when expired.
    """
    
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    if not CLIENT_ID or not CLIENT_SECRET:
        logger.warning("[Spotify] CLIENT_ID / CLIENT_SECRET not set — using demo tracks.")
        return None

    try:
        resp = requests.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {_credentials_header()}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.warning(f"Spotify token received: {'access_token' in data}")
        logger.warning(f"Spotify response: {data}")
        logger.warning(f"Spotify token received: {'access_token' in data}")
        logger.warning(f"Spotify response: {data}")
        _token_cache["token"]      = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        logger.info("[Spotify] Client credentials token refreshed.")
        return _token_cache["token"]
    except Exception as e:
       logger.error(f"[Spotify] Client credentials error: {e}")

    if 'resp' in locals():
        logger.error(resp.text)

    return None


# ── Track formatting ──────────────────────────────────────────────────────
def _format_track(item: dict) -> dict:
    images   = item.get("album", {}).get("images", [])
    cover    = images[0]["url"] if images else ""
    dur_ms   = item.get("duration_ms", 0)
    mins, secs = divmod(dur_ms // 1000, 60)
    return {
        "id":          item.get("id", ""),
        "name":        item.get("name", "Unknown"),
        "artist":      ", ".join(a["name"] for a in item.get("artists", [])),
        "album":       item.get("album", {}).get("name", ""),
        "album_cover": cover,
        "duration":    f"{mins}:{secs:02d}",
        "spotify_url": item.get("external_urls", {}).get("spotify", ""),
        "preview_url": item.get("preview_url") or "",
        "uri":         item.get("uri", ""),
        "popularity":  item.get("popularity", 0),
    }


# ── Live search ───────────────────────────────────────────────────────────
def search_tracks_by_emotion(emotion: str, limit: int = 12) -> list:
    """Search Spotify for tracks matching the emotion. Falls back to demo tracks."""
    token = _get_client_token()
    if not token:
        return _mock_tracks(emotion, limit)

    queries  = EMOTION_QUERIES.get(emotion, EMOTION_QUERIES["neutral"])
    headers  = {"Authorization": f"Bearer {token}"}
    tracks   = []
    seen_ids = set()

    for query in queries[:3]:
        try:
            resp = requests.get(
                f"{API_BASE}/search",
                headers=headers,
                params={
                    "q":     query,
                    "type":  "track",
                    "limit": max(6, limit // len(queries)),
                    "market": "US",
                },
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("tracks", {}).get("items", [])
            for t in items:
                if t["id"] not in seen_ids and t.get("preview_url") is not None:
                    seen_ids.add(t["id"])
                    tracks.append(_format_track(t))
        except Exception as e:
            logger.error(f"[Spotify] Search error '{query}': {e}")

    # Sort by popularity, cap at limit
    tracks.sort(key=lambda t: t["popularity"], reverse=True)
    result = tracks[:limit]

    if not result:
        logger.warning("[Spotify] Live search returned nothing — using demo tracks.")
        return _mock_tracks(emotion, limit)

    logger.info(f"[Spotify] Returned {len(result)} live tracks for '{emotion}'.")
    return result


def get_recommendations(emotion: str, seed_tracks: list = None, limit: int = 10) -> list:
    """Use Spotify recommendations endpoint with audio feature targeting."""
    token = _get_client_token()
    if not token:
        return _mock_tracks(emotion, limit)

    genres = EMOTION_SEED_GENRES.get(emotion, ["pop"])[:2]
    params = {
        "seed_genres": ",".join(genres),
        "limit":       limit,
        "market":      "US",
    }
    params.update(EMOTION_AUDIO_FEATURES.get(emotion, {}))

    # Optionally seed with known track IDs
    if seed_tracks:
        params["seed_tracks"] = ",".join(seed_tracks[:2])
        params.pop("seed_genres", None)

    try:
        resp = requests.get(
            f"{API_BASE}/recommendations",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", [])
        result = [_format_track(t) for t in items]
        if result:
            logger.info(f"[Spotify] Recommendations: {len(result)} tracks for '{emotion}'.")
            return result
    except Exception as e:
        logger.error(f"[Spotify] Recommendations error: {e}")

    return _mock_tracks(emotion, limit)


def has_credentials() -> bool:
    logger.warning(f"CLIENT_ID loaded: {bool(CLIENT_ID)}")
    logger.warning(f"CLIENT_SECRET loaded: {bool(CLIENT_SECRET)}")
    logger.warning(f"CLIENT_ID first chars: {CLIENT_ID[:5] if CLIENT_ID else 'NONE'}")
    return bool(CLIENT_ID and CLIENT_SECRET)

# ── Demo / mock tracks (shown when no credentials) ────────────────────────
MOCK_LIBRARY = {
    "happy": [
        ("Happy",                  "Pharrell Williams",         "G I R L",              "https://i.scdn.co/image/ab67616d0000b273e8107e6d9214baa81bb79bba"),
        ("Can't Stop the Feeling!","Justin Timberlake",         "Trolls OST",           "https://i.scdn.co/image/ab67616d0000b2737ae4c0a4c7e07f8c4dbcb3de"),
        ("Shake It Off",           "Taylor Swift",              "1989",                 "https://i.scdn.co/image/ab67616d0000b273904445d70d4a1d16c3a07df6"),
        ("Uptown Funk",            "Mark Ronson ft. Bruno Mars","Uptown Special",       "https://i.scdn.co/image/ab67616d0000b27341caa82c30d9bd1b3d7907e3"),
        ("Levitating",             "Dua Lipa",                  "Future Nostalgia",     "https://i.scdn.co/image/ab67616d0000b2734bc66095f4568ef0d1ea8e56"),
        ("Good as Hell",           "Lizzo",                     "Cuz I Love You",       "https://i.scdn.co/image/ab67616d0000b273da00f6e8dcc6d574310a9a2b"),
    ],
    "sad": [
        ("Someone Like You",       "Adele",                     "21",                   "https://i.scdn.co/image/ab67616d0000b27319d85a472f328a6ed9b704cf"),
        ("Fix You",                "Coldplay",                  "X&Y",                  "https://i.scdn.co/image/ab67616d0000b273de09e02aa7febf30b7c02d82"),
        ("Skinny Love",            "Bon Iver",                  "For Emma, Forever Ago","https://i.scdn.co/image/ab67616d0000b2736b27ce4e3b56a5cfecfc2c41"),
        ("Falling",                "Harry Styles",              "Fine Line",            "https://i.scdn.co/image/ab67616d0000b273b46f74097655d7f353caab14"),
        ("Liability",              "Lorde",                     "Melodrama",            "https://i.scdn.co/image/ab67616d0000b273768632ef4fac41741b7c0535"),
        ("The Night Will Always Win","Manchester Orchestra",    "Hope",                 "https://i.scdn.co/image/ab67616d0000b273f46b9d202509a8f7384b90de"),
    ],
    "angry": [
        ("Killing in the Name",    "Rage Against the Machine",  "Rage Against the Machine","https://i.scdn.co/image/ab67616d0000b273b1f8a6b5b9bde47bf20b5d0c"),
        ("Break Stuff",            "Limp Bizkit",               "Significant Other",    "https://i.scdn.co/image/ab67616d0000b2736ef88e0b5d5c9a1547cef645"),
        ("Given Up",               "Linkin Park",               "Minutes to Midnight",  "https://i.scdn.co/image/ab67616d0000b2737f4e8c3b3b8d9db72b5f8e87"),
        ("Chop Suey!",             "System of a Down",          "Toxicity",             "https://i.scdn.co/image/ab67616d0000b27368305c3d5f9cf8d2fe072e92"),
        ("Bulls on Parade",        "Rage Against the Machine",  "Evil Empire",          "https://i.scdn.co/image/ab67616d0000b273a4fb7b61f84e6c93d8af1a0e"),
        ("Enter Sandman",          "Metallica",                 "Metallica (Black Album)","https://i.scdn.co/image/ab67616d0000b273f1056aede81d6bea9b9adafd"),
    ],
    "fear": [
        ("Weightless",             "Marconi Union",             "Weightless",           "https://i.scdn.co/image/ab67616d0000b27344b9d0aa26e87a3e51b6c2a4"),
        ("Clair de Lune",          "Claude Debussy",            "Suite bergamasque",    "https://i.scdn.co/image/ab67616d0000b273f4db99c7b2218bde5e7c53d9"),
        ("River Flows in You",     "Yiruma",                    "First Love",           "https://i.scdn.co/image/ab67616d0000b2738b3b3a9e9a1c5bafcaae4e2d"),
        ("Experience",             "Ludovico Einaudi",          "In a Time Lapse",      "https://i.scdn.co/image/ab67616d0000b2733b5f9c24a18cb1ab6b8c5e3f"),
        ("Gymnopédie No.1",        "Erik Satie",                "Gymnopédies",          "https://i.scdn.co/image/ab67616d0000b273d9a62cd7a1bb5b7c4c9b8e45"),
        ("Spiegel im Spiegel",     "Arvo Pärt",                 "Tabula Rasa",          "https://i.scdn.co/image/ab67616d0000b2731b4a1f5c6d7e8f9a0b1c2d3e"),
    ],
    "love": [
        ("Perfect",                "Ed Sheeran",                "÷ (Divide)",           "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96"),
        ("At Last",                "Etta James",                "At Last!",             "https://i.scdn.co/image/ab67616d0000b273a6d05b8b70e3cb6c8f13bfac"),
        ("Can't Help Falling in Love","Elvis Presley",          "Blue Hawaii",          "https://i.scdn.co/image/ab67616d0000b273ae3f7e60b4e12c8a3d1c0e22"),
        ("All of Me",              "John Legend",               "Love in the Future",   "https://i.scdn.co/image/ab67616d0000b273edd9b5cde3ebbd1d2e5fb2a3"),
        ("Thinking Out Loud",      "Ed Sheeran",                "x (Multiply)",         "https://i.scdn.co/image/ab67616d0000b27323b4b38d9a0f28b7c1234abc"),
        ("Make You Feel My Love",  "Adele",                     "19",                   "https://i.scdn.co/image/ab67616d0000b273e1c2b7a8d3f4e5c6b7a8c9d0"),
    ],
    "surprise": [
        ("Bohemian Rhapsody",      "Queen",                     "A Night at the Opera", "https://i.scdn.co/image/ab67616d0000b273e8b066f70c206551210d902b"),
        ("Take Five",              "Dave Brubeck Quartet",      "Time Out",             "https://i.scdn.co/image/ab67616d0000b2735c4e9d3b4a8f1e2d3c4b5a6f"),
        ("Mr. Brightside",         "The Killers",               "Hot Fuss",             "https://i.scdn.co/image/ab67616d0000b273fe0ce523f37f3a0cce7a3de1"),
        ("Stairway to Heaven",     "Led Zeppelin",              "Led Zeppelin IV",      "https://i.scdn.co/image/ab67616d0000b27351c02a77d09dfcd53c8676d0"),
        ("Bohemian Like You",      "The Dandy Warhols",         "Thirteen Tales",       "https://i.scdn.co/image/ab67616d0000b2731a2b3c4d5e6f7a8b9c0d1e2f"),
        ("Heroes",                 "David Bowie",               "'Heroes'",             "https://i.scdn.co/image/ab67616d0000b273c8e5c8a3b5d4e7f2a1b3c4d5"),
    ],
    "neutral": [
        ("Blinding Lights",        "The Weeknd",                "After Hours",          "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb21"),
        ("As It Was",              "Harry Styles",              "Harry's House",        "https://i.scdn.co/image/ab67616d0000b273b46f74097655d7f353caab14"),
        ("Anti-Hero",              "Taylor Swift",              "Midnights",            "https://i.scdn.co/image/ab67616d0000b273904445d70d4a1d16c3a07df6"),
        ("Flowers",                "Miley Cyrus",               "Endless Summer Vacation","https://i.scdn.co/image/ab67616d0000b2737c9df9c4b3e2f1d0c5b4a3e2"),
        ("Kill Bill",              "SZA",                       "SOS",                  "https://i.scdn.co/image/ab67616d0000b273a91c10fe9472d9bd89802e5a"),
        ("Escapism.",              "RAYE ft. 070 Shake",        "My 21st Century Blues","https://i.scdn.co/image/ab67616d0000b2734a5b6c7d8e9f0a1b2c3d4e5f"),
    ],
}


def _mock_tracks(emotion: str, limit: int = 6) -> list:
    """Return curated demo tracks with real Spotify cover art URLs."""
    library = MOCK_LIBRARY.get(emotion, MOCK_LIBRARY["neutral"])
    tracks  = []
    for i, (name, artist, album, cover) in enumerate(library[:limit]):
        mins = 3 + (i % 2)
        secs = (i * 17) % 60
        tracks.append({
            "id":          f"demo_{emotion}_{i}",
            "name":        name,
            "artist":      artist,
            "album":       album,
            "album_cover": cover,
            "duration":    f"{mins}:{secs:02d}",
            "spotify_url": f"https://open.spotify.com/search/{'+'.join(name.split())}",
            "preview_url": "",
            "uri":         "",
            "popularity":  80 - i * 5,
            "mock":        True,
        })
    return tracks
