"""
tests/test_calibration.py
--------------------------
Run with: python tests/test_calibration.py path/to/Baby_Bieber.mp3

Validates audio feature formulas against known expected ranges for Baby by Justin Bieber.
If this test PASSES — formulas are calibrated and you can trust the pipeline.
If this test FAILS — feature formulas still need work.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.audio_analysis import AudioAnalyzer

# Expected ranges for Baby by Justin Bieber (ft. Ludacris)
# Source: Spotify Web API + music knowledge
EXPECTED = {
    "speechiness":       (0.05, 0.35),   # has heavy vocals but it's a song, not speech
    "instrumentalness":  (0.0,  0.20),   # very vocal-heavy, not instrumental
    "valence":           (0.60, 0.90),   # upbeat, happy, major key
    "loudness":          (-15,  -5),     # heavily mastered commercial track
    "danceability":      (0.65, 0.95),   # high danceability pop song
    "hook_arrival_second": (10, 30),     # hook (chorus) arrives by ~18s
}

TOLERANCE_NOTES = {
    "speechiness":       "Baby has strong lead vocals — should be low-mid (0.05-0.35)",
    "instrumentalness":  "Baby is a vocal pop song — should be near 0",
    "valence":           "Baby is major key happy pop — should be 0.6+",
    "loudness":          "Baby has -8 to -12 dBFS integrated loudness typically",
    "danceability":      "Baby has a driving pop beat — should be 0.65+",
    "hook_arrival_second": "Baby's first chorus hits at ~18s — hook should be within 30s",
}


def run_calibration(audio_path):
    print("=" * 60)
    print("  ComposerIQ — Feature Calibration Test")
    print("  Track: Baby by Justin Bieber")
    print("=" * 60)
    print(f"\nAnalyzing: {audio_path}\n")

    analyzer = AudioAnalyzer()
    result = analyzer.analyze(audio_path)

    print("\n--- Calibration Results ---\n")
    failures = []
    passes = []

    for key, (low, high) in EXPECTED.items():
        val = result.get(key, None)
        if val is None:
            failures.append(f"MISSING  {key}: key not in result")
            continue

        ok = low <= float(val) <= high
        status = "PASS" if ok else "FAIL"
        note = TOLERANCE_NOTES.get(key, "")

        line = f"{status:5}  {key:25} got={float(val):.4f}  expected=[{low}, {high}]"
        if not ok:
            line += f"\n        ↳ {note}"
            failures.append(line)
        else:
            passes.append(line)

    for p in passes:
        print(p)
    if failures:
        print()
        for f in failures:
            print(f)

    print(f"\n--- Summary ---")
    print(f"PASSED: {len(passes)}/{len(EXPECTED)}")
    print(f"FAILED: {len(failures)}/{len(EXPECTED)}")

    if not failures:
        print("\n✓ ALL CALIBRATION TESTS PASSED — formulas are working correctly!")
    else:
        print("\n✗ CALIBRATION FAILED — feature formulas need further adjustment")

    return failures


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_calibration.py path/to/Baby_Bieber.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    failures = run_calibration(audio_path)
    sys.exit(0 if not failures else 1)
