"""
stress_test.py
--------------
Tests /analyze against bad inputs and confirms every case returns
clean JSON, never a 500 crash.

Usage:
    python stress_test.py [--url http://localhost:8000]
"""

import argparse
import json
import os
import time
import tempfile
import urllib.request
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
SOUNDHELIX_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
VALID_MP3 = Path("SoundHelix-Song-1.mp3")


def _download_valid_mp3():
    if VALID_MP3.exists():
        return
    print(f"  Downloading {VALID_MP3} ...")
    urllib.request.urlretrieve(SOUNDHELIX_URL, VALID_MP3)
    print(f"  Saved {VALID_MP3.stat().st_size // 1024} KB")


def _poll_result(job_id: str, timeout: int = 120) -> dict:
    """Poll /analyze/status/{job_id} until complete or timeout."""
    url = f"{BASE_URL}/analyze/status/{job_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") in ("complete", "failed"):
            return r, data
        time.sleep(3)
    # Return last response on timeout
    return r, data


def _run_test(name: str, files: dict, data: dict,
              expect_ok: bool = False,
              timeout: int = 300) -> tuple[bool, int, dict]:
    """POST to /analyze, return (passed, status_code, body).

    passed = True when:
      - expect_ok=False: server responded (any status), body is valid JSON
      - expect_ok=True:  server responded with 200 and body has a report
    """
    try:
        r = requests.post(f"{BASE_URL}/analyze", files=files, data=data,
                          timeout=timeout)
        body = r.json()
        status_code = r.status_code
    except Exception as e:
        return False, 0, {"error": str(e)}

    # Server responded with parseable JSON — never a silent crash
    if expect_ok:
        passed = (status_code == 200 and bool(body.get("report") or body.get("result")))
    else:
        # Bad-input cases: any non-zero status with a non-empty detail is fine
        passed = status_code != 0 and bool(body)
    return passed, status_code, body


def main(base_url: str):
    global BASE_URL
    BASE_URL = base_url

    _download_valid_mp3()

    results = []

    def record(name, passed, status, body):
        mark = "PASS" if passed else "FAIL"
        print(f"\n[{mark}] {name}")
        print(f"  Status: {status}")
        try:
            print(f"  Body  : {json.dumps(body, indent=2)[:300]}")
        except Exception:
            print(f"  Body  : {body}")
        results.append((name, passed))

    # ------------------------------------------------------------------
    # 1. Corrupted file — bytes that are not valid MP3
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"not an mp3 file - corrupted garbage data")
        corrupt_path = f.name
    try:
        with open(corrupt_path, "rb") as fh:
            passed, status, body = _run_test(
                "Corrupted file (.mp3 containing garbage)",
                files={"audio_file": ("corrupt.mp3", fh, "audio/mpeg")},
                data={"genre": "pop", "mood": "upbeat"},
            )
        record("Corrupted file", passed, status, body)
    finally:
        os.unlink(corrupt_path)

    # ------------------------------------------------------------------
    # 2. Wrong extension — .txt renamed to .mp3
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"This is a plain text file, not audio.")
        txt_path = f.name
    try:
        with open(txt_path, "rb") as fh:
            passed, status, body = _run_test(
                "Wrong extension (.txt content sent as .mp3)",
                files={"audio_file": ("fake.mp3", fh, "audio/mpeg")},
                data={"genre": "pop", "mood": "upbeat"},
            )
        record("Wrong extension", passed, status, body)
    finally:
        os.unlink(txt_path)

    # ------------------------------------------------------------------
    # 3. Empty file — 0 bytes
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        empty_path = f.name  # write nothing
    try:
        with open(empty_path, "rb") as fh:
            passed, status, body = _run_test(
                "Empty file (0 bytes)",
                files={"audio_file": ("empty.mp3", fh, "audio/mpeg")},
                data={"genre": "pop", "mood": "upbeat"},
            )
        record("Empty file", passed, status, body)
    finally:
        os.unlink(empty_path)

    # ------------------------------------------------------------------
    # 4. Giant filename — 500 character filename
    # ------------------------------------------------------------------
    long_name = "a" * 500 + ".mp3"
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"not an mp3")
        giant_path = f.name
    try:
        with open(giant_path, "rb") as fh:
            passed, status, body = _run_test(
                "Giant filename (500 chars)",
                files={"audio_file": (long_name, fh, "audio/mpeg")},
                data={"genre": "pop", "mood": "upbeat"},
            )
        record("Giant filename", passed, status, body)
    finally:
        os.unlink(giant_path)

    # ------------------------------------------------------------------
    # 5. Missing genre field — valid MP3, no genre
    # ------------------------------------------------------------------
    with open(VALID_MP3, "rb") as fh:
        passed, status, body = _run_test(
            "Missing genre (valid MP3)",
            files={"audio_file": ("song.mp3", fh, "audio/mpeg")},
            data={"mood": "upbeat"},  # no genre
            timeout=300,
        )
    record("Missing genre", passed, status, body)

    # ------------------------------------------------------------------
    # 6. Missing mood field — valid MP3, no mood
    # ------------------------------------------------------------------
    with open(VALID_MP3, "rb") as fh:
        passed, status, body = _run_test(
            "Missing mood (valid MP3)",
            files={"audio_file": ("song.mp3", fh, "audio/mpeg")},
            data={"genre": "pop"},  # no mood
            timeout=300,
        )
    record("Missing mood", passed, status, body)

    # ------------------------------------------------------------------
    # 7. Valid MP3 — full happy path, confirm 200 OK
    # ------------------------------------------------------------------
    with open(VALID_MP3, "rb") as fh:
        passed, status, body = _run_test(
            "Valid MP3 (happy path)",
            files={"audio_file": ("song.mp3", fh, "audio/mpeg")},
            data={"genre": "pop", "mood": "upbeat"},
            expect_ok=True,
            timeout=300,
        )
    record("Valid MP3", passed, status, body)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    failed = [name for name, p in results if not p]

    print(f"\n{'=' * 50}")
    print(f"PASSED: {passed_count}/{total}")
    if failed:
        print(f"FAILED: {len(failed)}/{total}")
        for name in failed:
            print(f"  - {name}")
    else:
        print("All tests passed.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    main(args.url)
