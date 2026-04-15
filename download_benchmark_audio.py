"""
download_benchmark_audio.py
----------------------------
Downloads 90-second MP3 clips for each track in benchmark_manifest.json
using the yt-dlp Python API, then writes the local path back into audio_path.

Folder layout:
    benchmark_audio/
        cluster_0/
            <track_id>.mp3
        cluster_1/
            ...

Skips any track that already has a file at the expected output path.

Usage:
    python download_benchmark_audio.py [--manifest benchmark_manifest.json]
                                        [--out-dir benchmark_audio]
                                        [--cluster 0]
"""

import argparse
import json
import time
from pathlib import Path

import yt_dlp
import imageio_ffmpeg


# ---------------------------------------------------------------------------
# FFmpeg path (via imageio-ffmpeg if not on system PATH)
# ---------------------------------------------------------------------------

def _ffmpeg_path() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------

def _download_track(track_id: str, name: str, artist: str,
                    out_dir: Path) -> str | None:
    """
    Download the first YouTube result for '{name} {artist} audio',
    trim to 90 seconds, save as <track_id>.mp3 inside out_dir.
    Returns the absolute path on success, None on failure.
    """
    out_path = out_dir / f"{track_id}.mp3"
    if out_path.exists():
        return str(out_path)

    query = f"ytsearch1:{name} {artist} audio"

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "max_downloads": 1,
        "outtmpl": str(out_dir / f"{track_id}.%(ext)s"),
        "ffmpeg_location": _ffmpeg_path(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "5",
            }
        ],
        "postprocessor_args": {
            "ffmpeg": ["-ss", "0", "-t", "90"],
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
    except yt_dlp.utils.MaxDownloadsReached:
        pass
    except Exception:
        return None

    return str(out_path) if out_path.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(manifest_path: str, out_dir: str, cluster_filter: int | None) -> None:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    # Collect work items
    work = []
    for cluster_key, cluster in manifest.items():
        cluster_id = int(cluster_key.split("_")[-1])
        if cluster_filter is not None and cluster_id != cluster_filter:
            continue
        cluster_dir = Path(out_dir) / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for track in cluster["tracks"]:
            expected_path = cluster_dir / f"{track['track_id']}.mp3"
            if expected_path.exists():
                # Already downloaded — update manifest if audio_path is missing
                if not track.get("audio_path"):
                    track["audio_path"] = str(expected_path)
                continue
            work.append((cluster_key, cluster_id, track, cluster_dir))

    total = len(work)
    cluster_label = f"cluster {cluster_filter}" if cluster_filter is not None else "all clusters"
    print(f"Downloading {total} tracks for {cluster_label}")
    print()

    if total == 0:
        print("Nothing to do — all tracks already downloaded.")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return

    succeeded = 0
    failed = 0

    # Group by cluster for progress display
    cluster_totals: dict[int, int] = {}
    cluster_done: dict[int, int] = {}
    for _, cid, _, _ in work:
        cluster_totals[cid] = cluster_totals.get(cid, 0) + 1
        cluster_done[cid] = 0

    for cluster_key, cluster_id, track, cluster_dir in work:
        cluster_done[cluster_id] += 1
        done_str = f"{cluster_done[cluster_id]}/{cluster_totals[cluster_id]}"
        label = f"Cluster {cluster_id} [{done_str}] {track['name']} - {track['artist']}"
        print(label)

        path = _download_track(
            track_id=track["track_id"],
            name=track["name"],
            artist=track["artist"],
            out_dir=cluster_dir,
        )

        if path:
            succeeded += 1
            # Write path back into manifest in memory
            for t in manifest[cluster_key]["tracks"]:
                if t["track_id"] == track["track_id"]:
                    t["audio_path"] = path
                    break
        else:
            failed += 1
            print(f"  SKIP: no results for {track['name']} - {track['artist']}")

        # Persist manifest after each successful download
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Rate limiting
        time.sleep(2)

    print()
    print(f"Done. Downloaded: {succeeded}  Failed: {failed}  Total: {total}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmark_manifest.json")
    parser.add_argument("--out-dir",  default="benchmark_audio")
    parser.add_argument("--cluster",  type=int, default=None,
                        help="Download only this cluster ID (e.g. --cluster 0)")
    args = parser.parse_args()
    main(args.manifest, args.out_dir, args.cluster)
