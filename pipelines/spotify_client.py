import os
import random
import tempfile
import requests
import numpy as np
import librosa
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

SEARCH_QUERIES = [
    # Specific popular artists — guaranteed real popularity scores
    "Justin Bieber",
    "Taylor Swift",
    "Drake",
    "The Weeknd",
    "Billie Eilish",
    "Ed Sheeran",
    "Ariana Grande",
    "Post Malone",
    "Dua Lipa",
    "Harry Styles",
    # Chart / compilation searches
    "Billboard Hot 100",
    "Spotify top 50",
    "pop hits 2023",
    "pop hits 2022",
    "viral pop songs",
    "Grammy winners 2023",
    # Genre with real names
    "hip hop bangers",
    "RnB hits",
    "electronic dance hits",
]


def classify_mood(features: dict) -> str:
    energy = features.get("energy", 0.5)
    valence = features.get("valence", 0.5)
    tempo = features.get("tempo", 120)
    acousticness = features.get("acousticness", 0.5)
    danceability = features.get("danceability", 0.5)

    if energy > 0.7 and valence > 0.6 and danceability > 0.6:
        return "euphoric"
    elif energy > 0.7 and valence < 0.4:
        return "aggressive"
    elif energy < 0.4 and acousticness > 0.6 and valence < 0.5:
        return "melancholic"
    elif energy < 0.4 and valence > 0.6:
        return "peaceful"
    elif tempo > 140 and energy > 0.6:
        return "energetic"
    elif danceability > 0.7 and valence > 0.5:
        return "upbeat"
    elif acousticness > 0.7 and energy < 0.5:
        return "chill"
    else:
        return "neutral"


class SpotifyClient:
    def __init__(self):
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        if client_id and client_secret:
            response = requests.post(
                "https://accounts.spotify.com/api/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret
                }
            )
            if response.status_code == 200:
                access_token = response.json().get("access_token")
                self.sp = spotipy.Spotify(auth=access_token)
                print("Spotify: connected")
            else:
                auth = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=auth)
                print("Spotify: connected (fallback auth)")
        else:
            self.sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials()
            )

    def _fallback_features(self) -> dict:
        return {
            "danceability": random.uniform(0.3, 0.8),
            "energy": random.uniform(0.3, 0.8),
            "loudness": random.uniform(-15.0, -4.0),
            "speechiness": random.uniform(0.03, 0.1),
            "acousticness": random.uniform(0.1, 0.7),
            "instrumentalness": random.uniform(0.0, 0.4),
            "liveness": random.uniform(0.05, 0.3),
            "valence": random.uniform(0.3, 0.8),
            "tempo": random.uniform(85.0, 150.0),
        }

    def get_audio_features(self, preview_url: Optional[str]) -> dict:
        if not preview_url:
            return self._fallback_features()
        try:
            r = requests.get(preview_url, timeout=10)
            if r.status_code != 200:
                return self._fallback_features()

            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as f:
                f.write(r.content)
                tmp = f.name

            y, sr = librosa.load(tmp, duration=30)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            energy = float(np.mean(rms))
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            return {
                "danceability": float(np.clip(np.mean(zcr) * 10, 0, 1)),
                "energy": float(np.clip(energy * 100, 0, 1)),
                "loudness": float(np.mean(
                    librosa.amplitude_to_db(rms)
                )),
                "speechiness": float(np.clip(
                    np.mean(mfcc[1]) / 100 + 0.5, 0, 1
                )),
                "acousticness": float(np.clip(
                    1 - np.mean(centroid) / 8000, 0, 1
                )),
                "instrumentalness": float(np.clip(
                    1 - np.mean(mfcc[0]) / 100, 0, 1
                )),
                "liveness": float(np.clip(
                    np.std(rms) * 50, 0, 1
                )),
                "valence": float(np.clip(
                    np.mean(centroid) / 8000, 0, 1
                )),
                "tempo": float(tempo),
            }
        except Exception as e:
            print(f"    librosa failed: {e}, using fallback")
            return self._fallback_features()

    def crawl_broad_tracks(self, limit_per_query: int = 10) -> list:
        all_tracks = []
        seen_ids = set()

        for query in SEARCH_QUERIES:
            try:
                print(f"  Searching: {query}")
                results = self.sp.search(
                    q=query, type="track", limit=limit_per_query
                )
                items = results.get("tracks", {}).get("items", [])

                for item in items:
                    tid = item.get("id")
                    if not tid or tid in seen_ids:
                        continue
                    seen_ids.add(tid)

                    preview_url = item.get("preview_url")
                    features = self.get_audio_features(preview_url)
                    mood = classify_mood(features)

                    all_tracks.append({
                        "track_id": tid,
                        "name": item.get("name", "Unknown"),
                        "artist": item["artists"][0]["name"] if item.get("artists") else "Unknown",
                        "popularity": item.get("popularity", 0),
                        "preview_url": preview_url or "",
                        "search_query": query,
                        "mood": mood,
                        **features,
                    })

            except Exception as e:
                print(f"  Failed for '{query}': {e}")
                continue

        return all_tracks

    def get_features_as_vector(self, features: dict) -> Optional[list]:
        if not features:
            return None
        return [
            features.get("danceability", 0.0),
            features.get("energy", 0.0),
            features.get("loudness", 0.0) / 60.0 + 1.0,
            features.get("speechiness", 0.0),
            features.get("acousticness", 0.0),
            features.get("instrumentalness", 0.0),
            features.get("liveness", 0.0),
            features.get("valence", 0.0),
            features.get("tempo", 120.0) / 250.0,
        ]