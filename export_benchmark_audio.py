"""
export_benchmark_audio.py
-------------------------
Queries the Pinecone index, groups all stored tracks by cluster_id, and
writes benchmark_manifest.json.

NOTE: Pinecone metadata does NOT store audio_path or preview_url — those
are transient during ingestion and were never persisted. The manifest
therefore uses Spotify track IDs as identifiers. To get actual audio files
you have two options:
  a) Re-fetch preview URLs from Spotify using the track IDs, then download.
  b) Re-run ingestion with `ingest.py` modified to also store preview_url
     in Pinecone metadata.

Manifest structure:
  {
    "cluster_0": {
      "genre": "pop",
      "mood": "euphoric",
      "tracks": [
        {"track_id": "...", "name": "...", "artist": "...", "audio_path": null}
      ]
    },
    ...
  }

Usage:
    python export_benchmark_audio.py
"""

import os
import json
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def fetch_all_vectors(index):
    """
    Use index.list() to page through all vector IDs, then fetch in batches
    of 200 to retrieve metadata.
    """
    all_ids = []
    for id_batch in index.list():
        # list() yields either a list of IDs or a ListResponse object
        if hasattr(id_batch, 'vectors'):
            all_ids.extend(id_batch.vectors)
        elif isinstance(id_batch, list):
            all_ids.extend(id_batch)
        else:
            all_ids.extend(list(id_batch))

    print(f"Total vector IDs found: {len(all_ids)}")

    records = []
    batch_size = 200
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        response = index.fetch(ids=batch)
        vectors = response.get("vectors", {}) if isinstance(response, dict) else response.vectors
        for vid, data in vectors.items():
            meta = data.get("metadata", {}) if isinstance(data, dict) else data.metadata
            records.append({"track_id": vid, **(meta or {})})

    return records


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "composeriq-tracks")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    total = getattr(stats, 'total_vector_count', None) or stats.get('total_vector_count', 0)
    print(f"Index: {index_name}")
    print(f"Reported vector count: {total}")
    print()

    if total == 0:
        print("Index is empty — run ingest.py first.")
        return

    records = fetch_all_vectors(index)
    print(f"Fetched metadata for {len(records)} tracks\n")

    # Group by cluster_id
    by_cluster = defaultdict(list)
    cluster_meta = {}
    for r in records:
        cid = str(int(r.get("cluster_id", -1)))
        by_cluster[cid].append(r)
        if cid not in cluster_meta:
            cluster_meta[cid] = {
                "genre": r.get("genre", "unknown"),
                "mood":  r.get("mood", "neutral"),
            }

    # Print summary
    for cid in sorted(by_cluster, key=lambda x: int(x)):
        tracks = by_cluster[cid]
        label = cluster_meta[cid]
        print(f"Cluster {cid}  [{label['genre']} / {label['mood']}]  — {len(tracks)} tracks")
        for t in tracks:
            # audio_path is not stored; show what we have
            print(f"  track_id={t['track_id']}  name={t.get('name', '?')}  artist={t.get('artist', '?')}")
            print(f"    audio_path: <not stored in Pinecone — use track_id to re-fetch from Spotify>")
        print()

    # Build manifest
    manifest = {}
    for cid in sorted(by_cluster, key=lambda x: int(x)):
        key = f"cluster_{cid}"
        manifest[key] = {
            "genre": cluster_meta[cid]["genre"],
            "mood":  cluster_meta[cid]["mood"],
            "track_count": len(by_cluster[cid]),
            "tracks": [
                {
                    "track_id":   t["track_id"],
                    "name":       t.get("name", ""),
                    "artist":     t.get("artist", ""),
                    # audio_path is null; populate manually if you have local files.
                    # feature fields below come from Pinecone metadata and are used
                    # as proxies by build_cluster_stats.py when no audio is available.
                    "audio_path":    None,
                    "energy":        float(t.get("energy", 0.5)),
                    "danceability":  float(t.get("danceability", 0.5)),
                    "valence":       float(t.get("valence", 0.5)),
                    "tempo":         float(t.get("tempo", 120.0)),
                    "popularity":    int(t.get("popularity", 0)),
                }
                for t in by_cluster[cid]
            ],
        }

    out_path = "benchmark_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {out_path}")
    print(f"Total clusters: {len(manifest)}")
    print(f"Total tracks:   {sum(v['track_count'] for v in manifest.values())}")
    print()
    print("Next step: populate the audio_path fields in benchmark_manifest.json")
    print("  (download Spotify previews using each track_id, then run build_cluster_stats.py)")


if __name__ == "__main__":
    main()
