"""
Compute per-cluster calibration stats from benchmark_manifest.json.

Audio path resolution order for each track:
  1. audio_path field is set  → analyse the local file with MIRAnalyzer
  2. audio_path is null       → try Spotify preview_url via track_id
  3. No preview available     → fall back to feature-based proxies using
                                 the energy/danceability/valence/tempo fields
                                 that export_benchmark_audio.py stores in the
                                 manifest from Pinecone metadata.

Usage:
    python build_cluster_stats.py [--manifest benchmark_manifest.json] [--output cluster_stats.json]
"""

import argparse
import json
import os
import tempfile
import requests
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from data.mir_analyzer import MIRAnalyzer

load_dotenv()

_ANALYZER = MIRAnalyzer()


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def _spotify_token() -> str | None:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials",
                  "client_id": client_id,
                  "client_secret": client_secret},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception:
        return None


def _preview_url(track_id: str, token: str) -> str | None:
    try:
        resp = requests.get(
            f"https://api.spotify.com/v1/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return resp.json().get("preview_url")
    except Exception:
        return None


def _download_to_temp(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(resp.content)
            return f.name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature-based proxies (used when no audio file is reachable)
# ---------------------------------------------------------------------------

def _proxy_hook_time(energy: float, danceability: float) -> float:
    """
    Estimate hook arrival from engagement features.
    High energy + danceability → earlier hook (more immediate track).
    Maps to roughly 4–25 seconds.
    """
    engagement = energy * 0.55 + danceability * 0.45
    return float(np.clip(25.0 - engagement * 21.0, 4.0, 25.0))


def _proxy_raw_risk(energy: float, danceability: float,
                    valence: float, tempo: float) -> float:
    """
    Estimate raw skip risk (0–1) from stored features.
    Low energy, low danceability → high skip risk.
    """
    tempo_norm = float(np.clip(tempo / 150.0, 0.0, 1.0))
    return float(np.clip(
        1.0 - (energy * 0.40 + danceability * 0.35 + tempo_norm * 0.15 + valence * 0.10),
        0.0, 1.0,
    ))


# ---------------------------------------------------------------------------
# Per-track measurement
# ---------------------------------------------------------------------------

def _measure_track(t: dict, token: str | None) -> tuple[float, float] | None:
    """
    Returns (hook_time, raw_risk) for one track, or None if all paths fail.
    Tries: local file → Spotify preview → feature proxy.
    """
    name = t.get("name", t.get("track_id", "?"))
    audio_path = t.get("audio_path")
    tmp_path = None

    # --- Path 1: local file ---
    if audio_path:
        try:
            hook_time, raw_risk = _analyse_audio(audio_path)
            print(f"  OK  (local)   {name}")
            return hook_time, raw_risk
        except Exception as e:
            print(f"  WARN local file failed ({e}), trying Spotify")

    # --- Path 2: Spotify preview ---
    if token:
        preview = _preview_url(t["track_id"], token)
        if preview:
            tmp_path = _download_to_temp(preview)
            if tmp_path:
                try:
                    hook_time, raw_risk = _analyse_audio(tmp_path)
                    print(f"  OK  (preview) {name}")
                    return hook_time, raw_risk
                except Exception as e:
                    print(f"  WARN preview analysis failed ({e}), using proxy")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # --- Path 3: feature proxy ---
    energy      = float(t.get("energy",       0.5))
    danceability = float(t.get("danceability", 0.5))
    valence     = float(t.get("valence",       0.5))
    tempo       = float(t.get("tempo",         120.0))

    hook_time = _proxy_hook_time(energy, danceability)
    raw_risk  = _proxy_raw_risk(energy, danceability, valence, tempo)
    print(f"  OK  (proxy)   {name}  "
          f"[energy={energy:.2f} dance={danceability:.2f}]  "
          f"hook~{hook_time:.1f}s  risk~{raw_risk:.3f}")
    return hook_time, raw_risk


def _analyse_audio(path: str) -> tuple[float, float]:
    y = _ANALYZER.load(path)
    hook_time, _ = _ANALYZER._detect_hook(y)
    hook_time = hook_time or 30.0

    y30 = y[:_ANALYZER.sr * 30]
    rms = librosa.feature.rms(y=y30, hop_length=_ANALYZER.hop)[0]
    f10 = librosa.time_to_frames(10, sr=_ANALYZER.sr, hop_length=_ANALYZER.hop)
    slope = np.polyfit(np.arange(len(rms[:f10])), rms[:f10], 1)[0]
    centroid = librosa.feature.spectral_centroid(y=y30, sr=_ANALYZER.sr)[0]
    onset_env = librosa.onset.onset_strength(
        y=y30, sr=_ANALYZER.sr, hop_length=_ANALYZER.hop
    )
    factors = {
        'energy_slope':  float(np.clip(-slope * 1e4, 0, 1)),
        'hook_delay':    float(np.clip(hook_time / 30, 0, 1)),
        'spec_variance': float(np.clip(
            1 - np.std(centroid[:f10]) / (np.mean(centroid[:f10]) + 1e-8) * 5, 0, 1
        )),
        'onset_density': float(np.clip(
            1 - np.mean(onset_env > np.percentile(onset_env, 60)) * 3, 0, 1
        )),
    }
    raw_risk = (
        factors['energy_slope']  * 0.30 +
        factors['hook_delay']    * 0.35 +
        factors['spec_variance'] * 0.20 +
        factors['onset_density'] * 0.15
    )
    return hook_time, raw_risk


# ---------------------------------------------------------------------------
# Per-cluster aggregation
# ---------------------------------------------------------------------------

def build_stats_for_cluster(tracks: list, cluster_id: int,
                             token: str | None) -> dict:
    hook_times = []
    raw_risks  = []

    for t in tracks:
        result = _measure_track(t, token)
        if result is None:
            continue
        hook_times.append(result[0])
        raw_risks.append(result[1])

    if not hook_times:
        return {}

    return {
        'cluster_id':     cluster_id,
        'n_tracks':       len(hook_times),
        'hook_time_mean': round(float(np.mean(hook_times)), 3),
        'hook_time_std':  round(float(np.std(hook_times)) or 1.0, 3),
        'skip_risk_mean': round(float(np.mean(raw_risks)), 4),
        'skip_risk_std':  round(float(np.std(raw_risks)) or 0.01, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='benchmark_manifest.json')
    parser.add_argument('--output',   default='cluster_stats.json')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loaded manifest: {len(manifest)} clusters")
    token = _spotify_token()
    print(f"Spotify token: {'acquired' if token else 'unavailable — proxy mode only'}\n")

    all_stats = {}

    for cluster_key in sorted(manifest, key=lambda k: int(k.split('_')[-1])):
        entry = manifest[cluster_key]
        cluster_id = int(cluster_key.split('_')[-1])
        tracks = entry.get("tracks", [])
        genre  = entry.get("genre", "?")
        mood   = entry.get("mood",  "?")

        print(f"Cluster {cluster_id} [{genre} / {mood}] — {len(tracks)} tracks")
        stats = build_stats_for_cluster(tracks, cluster_id, token)
        if stats:
            all_stats[str(cluster_id)] = stats
            print(f"  -> hook_time mean={stats['hook_time_mean']:.2f}s  "
                  f"std={stats['hook_time_std']:.2f}")
            print(f"  -> skip_risk mean={stats['skip_risk_mean']:.4f}  "
                  f"std={stats['skip_risk_std']:.4f}")
        else:
            print(f"  -> No usable tracks — skipped")
        print()

    with open(args.output, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"Saved {len(all_stats)} cluster(s) to {args.output}")
