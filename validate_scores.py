"""
Sanity-check that the 4-layer scoring pipeline direction is correct.

Usage (your own files):
    python validate_scores.py early.mp3 delayed.mp3 hienergy.mp3

Usage (auto-download free samples):
    python validate_scores.py

Assertions:
    early-hook track   → hook_percentile      > 50
    delayed-hook track → hook_percentile      < 40
    high-energy track  → skip_risk_percentile < 50
    early-hook track   → engagement_score     > delayed-hook engagement_score
"""

import os
import sys
import tempfile
import requests
from dataclasses import dataclass
from pipelines.audio_analysis import AudioAnalyzer
from data.mir_analyzer import MIRAnalyzer, MIRResult
from pipelines.scoring_engine import ScoringEngine, ScoredTrack

# Default cluster stats
_STATS = {'skip_risk_mean': 0.13, 'skip_risk_std': 0.10}

# Free sample URLs chosen for their audio characteristics:
#   Song-1  – electronic dance, hits full energy within the first 2-3 s  (early hook)
#   Song-13 – slow-building ambient/new-age intro, musical content delayed (delayed hook)
#   Song-7  – up-tempo high-BPM track, dense onsets throughout            (high energy)
_DEFAULT_URLS = {
    "early_hook":   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "delayed_hook": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-13.mp3",
    "high_energy":  "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
}

LABELS = ["early_hook", "delayed_hook", "high_energy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, label: str) -> str:
    print(f"  Downloading {label} ... ", end="", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(r.content)
        path = f.name
    print(f"{len(r.content) // 1024} KB  ->  {path}")
    return path


def _resolve_paths(argv: list[str]) -> tuple[list[str], list[bool]]:
    """Return (paths, should_cleanup) for the 3 tracks."""
    if len(argv) >= 3:
        paths = argv[:3]
        return paths, [False, False, False]

    print("No paths supplied — downloading free sample tracks ...\n")
    paths, cleanup = [], []
    for label in LABELS:
        p = _download(_DEFAULT_URLS[label], label)
        paths.append(p)
        cleanup.append(True)
    print()
    return paths, cleanup


@dataclass
class Result:
    label: str
    path: str
    mir: MIRResult
    scored: ScoredTrack


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(paths: list[str]) -> list[Result]:
    audio_analyzer = AudioAnalyzer()
    mir_analyzer = MIRAnalyzer()
    scoring_engine = ScoringEngine()
    results = []
    for label, path in zip(LABELS, paths):
        print(f"Analyzing [{label}]  {os.path.basename(path)} ...")
        # Layer 1
        features = audio_analyzer.analyze(path)
        # Layer 2
        mir = mir_analyzer.analyze(features, _STATS, genre="electronic")
        # Layer 3
        scored = scoring_engine.score(features, mir)
        results.append(Result(label=label, path=path, mir=mir, scored=scored))
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

COL = 22

def _row(key: str, *values) -> str:
    return f"  {key:<20}" + "".join(f"{str(v):<{COL}}" for v in values)


def print_table(results: list[Result]) -> None:
    print()
    print("=" * (20 + COL * 3 + 2))
    print(_row("", *[r.label for r in results]))
    print("=" * (20 + COL * 3 + 2))
    keys = [
        ("hook_time",            lambda r: f"{r.mir.hook_time}s" if r.mir.hook_time else "None"),
        ("hook_percentile",      lambda r: r.mir.hook_percentile),
        ("hook_confidence",      lambda r: r.mir.hook_confidence),
        ("skip_risk_pct",        lambda r: r.mir.skip_risk_percentile),
        ("raw_risk",             lambda r: r.mir.raw_risk),
        ("engagement",           lambda r: r.scored.engagement_score),
        ("spotifycore",          lambda r: r.scored.spotifycore_score),
        ("energy_slope",         lambda r: r.mir.factor_breakdown.get("energy_slope", "?")),
        ("hook_delay",           lambda r: r.mir.factor_breakdown.get("hook_delay", "?")),
        ("spec_variance",        lambda r: r.mir.factor_breakdown.get("spec_variance", "?")),
        ("onset_density",        lambda r: r.mir.factor_breakdown.get("onset_density", "?")),
    ]
    for name, getter in keys:
        print(_row(name, *[getter(r) for r in results]))
    print("-" * (20 + COL * 3 + 2))
    for r in results:
        comps = r.scored.components
        print(f"  {r.label:<20} signals: " + ", ".join(f"{k}={v}" for k, v in comps.items()))
    print("=" * (20 + COL * 3 + 2))
    print()


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def check(results: list[Result]) -> bool:
    by_label = {r.label: r for r in results}
    early   = by_label["early_hook"]
    delayed = by_label["delayed_hook"]
    hienrgy = by_label["high_energy"]

    checks = [
        (
            "early_hook   hook_percentile > 50",
            early.mir.hook_percentile > 50,
            f"got {early.mir.hook_percentile}",
        ),
        (
            "delayed_hook hook_percentile < 40",
            delayed.mir.hook_percentile < 40,
            f"got {delayed.mir.hook_percentile}",
        ),
        (
            "high_energy  skip_risk_pct < 50",
            hienrgy.mir.skip_risk_percentile < 50,
            f"got {hienrgy.mir.skip_risk_percentile}",
        ),
        (
            "early_hook engagement > delayed_hook engagement",
            early.scored.engagement_score > delayed.scored.engagement_score,
            f"got {early.scored.engagement_score} vs {delayed.scored.engagement_score}",
        ),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}  ({detail})")
        if not passed:
            all_pass = False

    print()
    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    paths, cleanup = _resolve_paths(sys.argv[1:])

    try:
        results = run(paths)
        print_table(results)
        print("Assertions:")
        passed = check(results)
        if passed:
            print("All assertions passed. Scoring direction is correct.")
            sys.exit(0)
        else:
            print("One or more assertions failed — see table above to diagnose.")
            print("Swap in a different sample for the failing category if the")
            print("downloaded track does not match its expected characteristic.")
            sys.exit(1)
    finally:
        for path, should in zip(paths, cleanup):
            if should:
                try:
                    os.unlink(path)
                except OSError:
                    pass
