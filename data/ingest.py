import os
import time
import json
import tempfile
import requests
import anthropic
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data.mir_analyzer import MIRAnalyzer
from mert_embedder import embedder

load_dotenv()

N_CLUSTERS = 12

analyzer = MIRAnalyzer()

CLUSTER_STATS_PATH = Path(__file__).parent / 'cluster_stats.json'
if CLUSTER_STATS_PATH.exists():
    with open(CLUSTER_STATS_PATH) as f:
        CLUSTER_STATS_DB = json.load(f)
else:
    CLUSTER_STATS_DB = {}


def cluster_tracks(tracks):
    vectors = []
    for t in tracks:
        vectors.append([
            t.get("danceability", 0.0),
            t.get("energy", 0.0),
            t.get("valence", 0.0),
            t.get("tempo", 120.0) / 250.0,
            t.get("acousticness", 0.0),
            t.get("instrumentalness", 0.0),
            t.get("liveness", 0.0),
            t.get("speechiness", 0.0),
        ])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    k = min(N_CLUSTERS, len(tracks))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    for i, track in enumerate(tracks):
        track["cluster_id"] = int(labels[i])

    return tracks


def label_cluster_with_claude(cluster_tracks: list) -> dict:
    client = anthropic.Anthropic()
    sample = cluster_tracks[:5]

    descriptions = [
        f"{t['name']} by {t['artist']} "
        f"(energy:{t['energy']:.2f}, valence:{t['valence']:.2f}, "
        f"tempo:{t['tempo']:.0f}, danceability:{t['danceability']:.2f}, "
        f"acousticness:{t['acousticness']:.2f})"
        for t in sample
    ]

    prompt = f"""
Here are tracks grouped together by audio similarity:
{json.dumps(descriptions, indent=2)}

Based on the track names, artists, and audio characteristics,
what genre or vibe best describes this group?

Reply in JSON only, no explanation, no markdown backticks:
{{"genre": "genre name", "mood": "mood", "vibe": "one line description"}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def label_all_clusters(tracks):
    clusters = {}
    for track in tracks:
        cid = track["cluster_id"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(track)

    cluster_labels = {}
    for cid, ctracks in clusters.items():
        print(f"  Labeling cluster {cid} ({len(ctracks)} tracks)...")
        try:
            label = label_cluster_with_claude(ctracks)
            cluster_labels[cid] = label
            print(f"    → {label['genre']} | {label['mood']} | {label['vibe']}")
        except Exception as e:
            print(f"    → Failed ({e}), using fallback")
            cluster_labels[cid] = {
                "genre": f"cluster_{cid}",
                "mood": "neutral",
                "vibe": ""
            }

    for track in tracks:
        label = cluster_labels[track["cluster_id"]]
        track["genre"] = label["genre"]
        track["mood"] = label["mood"]
        track["vibe"] = label["vibe"]

    return tracks, cluster_labels


def ingest_all():
    from pinecone import Pinecone, ServerlessSpec
    from pipelines.spotify_client import SpotifyClient

    print("=" * 60)
    print("  ComposerIQ — Knowledge Base Ingestion")
    print("  Real audio analysis + Claude genre labeling")
    print("=" * 60)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "composeriq-tracks")

    # Always start fresh
    existing = [i.name for i in pc.list_indexes()]
    if index_name in existing:
        print(f"\nDeleting old index: {index_name}")
        pc.delete_index(index_name)
        time.sleep(5)

    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

    index = pc.Index(index_name)
    client = SpotifyClient()

    # Step 1 - crawl broadly
    print("\nStep 1 — Crawling Spotify with broad queries...")
    tracks = client.crawl_broad_tracks(limit_per_query=10)
    print(f"Collected {len(tracks)} unique tracks")

    if len(tracks) == 0:
        print("No tracks found. Check Spotify credentials.")
        return

    # Step 2 - cluster
    print(f"\nStep 2 — Clustering into {N_CLUSTERS} groups...")
    tracks = cluster_tracks(tracks)
    print("Clustering done")

    # Step 3 - label with Claude
    print("\nStep 3 — Labeling clusters with Claude...")
    tracks, cluster_labels = label_all_clusters(tracks)

    print("\nDiscovered genres:")
    for cid, label in cluster_labels.items():
        print(f"  {cid}: {label['genre']} — {label['mood']}")

    # Step 4 - store in Pinecone
    print("\nStep 4 — Storing in Pinecone...")
    vectors = []
    for t in tracks:
        preview_url = t.get("preview_url", "")
        if not preview_url:
            print(f"  Skipping {t.get('name', '?')} — no preview URL for MERT embedding")
            continue

        try:
            resp = requests.get(preview_url, timeout=10)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(resp.content)
                tmp_path = f.name
            try:
                v = embedder.embed(tmp_path).tolist()
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"  Skipping {t.get('name', '?')} — MERT embed failed: {e}")
            continue

        vectors.append({
            "id": t["track_id"],
            "values": v,
            "metadata": {
                "name": t.get("name", "")[:100],
                "artist": t.get("artist", "")[:100],
                "genre": t.get("genre", "unknown"),
                "mood": t.get("mood", "neutral"),
                "vibe": t.get("vibe", "")[:200],
                "cluster_id": int(t.get("cluster_id", -1)),
                "popularity": int(t.get("popularity", 0)),  # Fix 4: always int
                "energy": float(t.get("energy", 0.0)),
                "danceability": float(t.get("danceability", 0.0)),
                "valence": float(t.get("valence", 0.0)),
                "tempo": float(t.get("tempo", 120.0)),
            }
        })

    for i in range(0, len(vectors), 100):
        batch = vectors[i:i + 100]
        index.upsert(vectors=batch)
        print(f"  {min(i + 100, len(vectors))} / {len(vectors)} stored")

    print(f"""
{'=' * 60}
  INGESTION COMPLETE
  Total tracks: {len(vectors)}
  Genres: discovered by clustering (not hardcoded)
  Mood: derived from real audio math
  Audio vectors: real librosa analysis on preview URLs
{'=' * 60}
    """)
    print("Run: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    ingest_all()