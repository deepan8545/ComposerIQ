"""
migrate_to_mert.py
------------------
Migrates the legacy 9-dimensional Spotify-feature Pinecone index to a new
768-dimensional MERT-embedding index.

Safety guarantees
-----------------
* The OLD index (composeriq-tracks, 9D) is NEVER deleted or modified.
* The NEW index (composeriq-mert-768, 768D) is created fresh if it does not
  exist, or reused if it already exists (upserts are idempotent).
* Each track is processed independently; a failure on one track does not
  affect any other.

Audio resolution order for each track
--------------------------------------
1. Local benchmark_audio/<cluster_id>/<track_id>.mp3 (from download_benchmark_audio.py)
2. Spotify preview URL (fetched live via track_id using credentials in .env)
3. If neither resolves  →  track is skipped and logged.
   MERT requires real audio; there is no meaningful vector-space fallback.

Usage
-----
    python migrate_to_mert.py [--dry-run]
                               [--audio-dir benchmark_audio]

    --dry-run   Reads old index and checks which tracks have reachable audio
                without writing anything to the new index.
    --audio-dir Path to local benchmark audio directory (default: benchmark_audio)

Environment variables required
--------------------------------
    PINECONE_API_KEY
    PINECONE_INDEX_NAME          (old 9D index, default: composeriq-tracks)
    PINECONE_MERT_INDEX_NAME     (new 768D index, default: composeriq-mert-768)
    SPOTIFY_CLIENT_ID            (optional — only used if local audio missing)
    SPOTIFY_CLIENT_SECRET        (optional — only used if local audio missing)
"""

import argparse
import os
import sys
import time
import tempfile
import requests
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from mert_embedder import embedder

load_dotenv()

OLD_INDEX_DEFAULT  = "composeriq-tracks"
NEW_INDEX_DEFAULT  = "composeriq-mert-768"
NEW_DIMENSION      = 768
UPSERT_BATCH_SIZE  = 10   # small batches — MERT is heavy
PROGRESS_EVERY     = 10
AUDIO_DIR_DEFAULT  = "benchmark_audio"


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def _spotify_token() -> str | None:
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    sec = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not sec:
        return None
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials",
                  "client_id": cid, "client_secret": sec},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["access_token"]
    except Exception as e:
        print(f"[WARN] Spotify token failed: {e}")
        return None


def _get_preview_url(track_id: str, token: str) -> str | None:
    try:
        r = requests.get(
            f"https://api.spotify.com/v1/tracks/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return r.json().get("preview_url")
    except Exception:
        return None


def _download_to_temp(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(r.content)
            return f.name
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _fetch_all_records(index) -> list[dict]:
    """
    Page through index.list() to collect all IDs, then fetch metadata
    in batches of 200.
    """
    all_ids = []
    for page in index.list():
        if hasattr(page, "vectors"):
            all_ids.extend(page.vectors)
        elif isinstance(page, list):
            all_ids.extend(page)
        else:
            all_ids.extend(list(page))

    print(f"  Found {len(all_ids)} vector IDs in old index")

    records = []
    for i in range(0, len(all_ids), 200):
        batch = all_ids[i:i + 200]
        resp = index.fetch(ids=batch)
        vecs = resp.get("vectors", {}) if isinstance(resp, dict) else resp.vectors
        for vid, data in vecs.items():
            meta = data.get("metadata", {}) if isinstance(data, dict) else data.metadata
            records.append({"track_id": vid, **(meta or {})})

    return records


def _ensure_new_index(pc: Pinecone, name: str) -> object:
    existing = {idx.name for idx in pc.list_indexes()}
    if name in existing:
        print(f"  New index '{name}' already exists — reusing")
    else:
        print(f"  Creating new index '{name}' (dimension={NEW_DIMENSION}, metric=cosine) ...")
        pc.create_index(
            name=name,
            dimension=NEW_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("  Waiting for index to be ready ...")
        time.sleep(10)
    return pc.Index(name)


# ---------------------------------------------------------------------------
# Per-track embedding
# ---------------------------------------------------------------------------

def _find_local_audio(track_id: str, audio_dir: str,
                      cluster_id: str | None = None) -> str | None:
    """
    Looks for <audio_dir>/cluster_<id>/<track_id>.mp3 across all clusters.
    If cluster_id is provided, checks that cluster first.
    """
    base = os.path.abspath(audio_dir)
    if not os.path.isdir(base):
        return None
    # Try supplied cluster_id first
    if cluster_id is not None:
        candidate = os.path.join(base, f"cluster_{cluster_id}", f"{track_id}.mp3")
        if os.path.isfile(candidate):
            return candidate
    # Scan all cluster subdirectories
    for cluster_dir in sorted(os.listdir(base)):
        candidate = os.path.join(base, cluster_dir, f"{track_id}.mp3")
        if os.path.isfile(candidate):
            return candidate
    return None


def _embed_track(track_id: str, token: str | None,
                 audio_dir: str = AUDIO_DIR_DEFAULT,
                 cluster_id: str | None = None) -> list[float] | None:
    """
    Returns a 768-dim MERT vector, or None if audio is unreachable.
    Resolution order:
      1. Local benchmark_audio/<cluster>/<track_id>.mp3
      2. Spotify preview URL (if token available)
    """
    tmp_to_delete = None
    try:
        # 1. Local file
        local_path = _find_local_audio(track_id, audio_dir, cluster_id)
        if local_path:
            try:
                return embedder.embed(local_path).tolist()
            except Exception as e:
                print(f"    [MERT ERROR] {track_id} (local): {e}")
                return None

        # 2. Spotify preview fallback
        if not token:
            return None
        preview_url = _get_preview_url(track_id, token)
        if not preview_url:
            return None
        tmp_to_delete = _download_to_temp(preview_url)
        if not tmp_to_delete:
            return None
        try:
            return embedder.embed(tmp_to_delete).tolist()
        except Exception as e:
            print(f"    [MERT ERROR] {track_id} (spotify): {e}")
            return None
    finally:
        if tmp_to_delete:
            try:
                os.unlink(tmp_to_delete)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(old_index_name: str, new_index_name: str, dry_run: bool,
            audio_dir: str = AUDIO_DIR_DEFAULT) -> None:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Verify old index exists
    existing = {idx.name for idx in pc.list_indexes()}
    if old_index_name not in existing:
        sys.exit(f"ERROR: Old index '{old_index_name}' not found. Check PINECONE_INDEX_NAME.")

    old_index = pc.Index(old_index_name)
    old_stats = old_index.describe_index_stats()
    old_dim   = getattr(old_stats, "dimension", "?")
    old_count = getattr(old_stats, "total_vector_count", "?")
    print(f"\nOld index : {old_index_name}  (dim={old_dim}, vectors={old_count})")

    # Prepare new index (skip creation in dry-run)
    if dry_run:
        print(f"New index : {new_index_name}  [DRY RUN — not created]")
        new_index = None
    else:
        print(f"New index : ", end="")
        new_index = _ensure_new_index(pc, new_index_name)
        new_stats  = new_index.describe_index_stats()
        new_count  = getattr(new_stats, "total_vector_count", 0)
        print(f"  Already contains {new_count} vectors")

    # Fetch all metadata from old index
    print(f"\nReading all records from '{old_index_name}' ...")
    records = _fetch_all_records(old_index)
    print(f"  Fetched metadata for {len(records)} tracks\n")

    # Acquire Spotify token
    token = _spotify_token()
    if token:
        print("Spotify token acquired\n")
    else:
        print("[WARN] No Spotify credentials — all tracks will be skipped\n")

    # Process tracks
    succeeded, skipped_no_audio, skipped_mert_error = [], [], []
    pending_upsert = []

    for i, rec in enumerate(records, start=1):
        track_id = rec["track_id"]
        name     = rec.get("name", track_id)

        vec = _embed_track(track_id, token,
                           audio_dir=audio_dir,
                           cluster_id=str(rec.get("cluster_id", "")))

        if vec is None:
            skipped_no_audio.append(track_id)
            status = "SKIP (no audio)"
        else:
            succeeded.append(track_id)
            status = "OK"
            if not dry_run:
                pending_upsert.append({
                    "id":     track_id,
                    "values": vec,
                    "metadata": {
                        k: rec[k] for k in (
                            "name", "artist", "genre", "mood", "vibe",
                            "cluster_id", "popularity", "energy",
                            "danceability", "valence", "tempo",
                        ) if k in rec
                    },
                })

        if i % PROGRESS_EVERY == 0 or i == len(records):
            print(f"  [{i:>3}/{len(records)}]  {status:<20}  {name[:50]}")

        # Flush upsert batch
        if not dry_run and len(pending_upsert) >= UPSERT_BATCH_SIZE:
            new_index.upsert(vectors=pending_upsert)
            pending_upsert.clear()

    # Final flush
    if not dry_run and pending_upsert:
        new_index.upsert(vectors=pending_upsert)

    # Summary
    print(f"""
{'=' * 60}
  MIGRATION {'(DRY RUN) ' if dry_run else ''}COMPLETE
  Old index : {old_index_name}  ({len(records)} tracks, UNCHANGED)
  New index : {new_index_name}
  Migrated  : {len(succeeded)} tracks
  Skipped   : {len(skipped_no_audio)} (no Spotify preview available)
  MERT errors: {len(skipped_mert_error)}
{'=' * 60}
""")

    if skipped_no_audio:
        print("Tracks skipped (no audio) — Spotify preview_url was null:")
        for tid in skipped_no_audio[:20]:
            print(f"  {tid}")
        if len(skipped_no_audio) > 20:
            print(f"  ... and {len(skipped_no_audio) - 20} more")

    if not dry_run and len(succeeded) > 0:
        print(f"\nNext step: set PINECONE_MERT_INDEX_NAME={new_index_name} in .env")
        print("The retrieval layer will now prefer the MERT index automatically.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check audio reachability without writing to the new index")
    parser.add_argument("--old-index", default=os.getenv("PINECONE_INDEX_NAME", OLD_INDEX_DEFAULT))
    parser.add_argument("--new-index", default=os.getenv("PINECONE_MERT_INDEX_NAME", NEW_INDEX_DEFAULT))
    parser.add_argument("--audio-dir", default=AUDIO_DIR_DEFAULT,
                        help="Local benchmark audio directory (default: benchmark_audio)")
    args = parser.parse_args()

    migrate(args.old_index, args.new_index, args.dry_run, audio_dir=args.audio_dir)
