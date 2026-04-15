"""
Integration test for MIRAnalyzer (Layer 2) + ScoringEngine (Layer 3).
Downloads a free public MP3, runs the full pipeline, and validates the result shape.
"""
import os
import tempfile
import requests
from pipelines.audio_analysis import AudioAnalyzer
from data.mir_analyzer import MIRAnalyzer, MIRResult
from pipelines.scoring_engine import ScoringEngine, ScoredTrack

MP3_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"


def test_mir_analyzer():
    print(f"Downloading test audio from {MP3_URL} ...")
    resp = requests.get(MP3_URL, timeout=30)
    resp.raise_for_status()
    print(f"Downloaded {len(resp.content) / 1024:.1f} KB")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    try:
        # Layer 1: Feature extraction
        audio_analyzer = AudioAnalyzer()
        features = audio_analyzer.analyze(tmp_path)
        print("\n--- Layer 1: Features ---")
        for k, v in features.items():
            if k != 'energy_curve':
                print(f"  {k}: {v}")

        # Layer 2: MIR percentiles
        mir_analyzer = MIRAnalyzer()
        cluster_stats = {
            'skip_risk_mean': 0.13,
            'skip_risk_std': 0.10,
        }
        result = mir_analyzer.analyze(features, cluster_stats, genre="electronic")

        print("\n--- Layer 2: MIR Percentiles ---")
        print(f"  hook_time:             {result.hook_time}")
        print(f"  hook_confidence:       {result.hook_confidence}")
        print(f"  hook_percentile:       {result.hook_percentile}")
        print(f"  skip_risk_percentile:  {result.skip_risk_percentile}")
        print(f"  raw_risk:              {result.raw_risk}")
        print(f"  cluster_id:            {result.cluster_id}")
        print(f"  genre_tier:            {result.genre_tier}")
        print(f"  factor_breakdown:")
        for k, v in result.factor_breakdown.items():
            print(f"    {k}: {v}")

        assert isinstance(result, MIRResult), "Result is not a MIRResult instance"
        assert 0 <= result.hook_percentile <= 100, (
            f"hook_percentile out of range: {result.hook_percentile}"
        )
        assert 0 <= result.skip_risk_percentile <= 100, (
            f"skip_risk_percentile out of range: {result.skip_risk_percentile}"
        )
        assert result.hook_time is None or (
            isinstance(result.hook_time, float) and result.hook_time > 0
        ), f"hook_time invalid: {result.hook_time}"

        # Layer 3: Scoring engine
        scoring_engine = ScoringEngine()
        scored = scoring_engine.score(features, result)

        print("\n--- Layer 3: Calibrated Scores ---")
        print(f"  engagement_score:  {scored.engagement_score}")
        print(f"  spotifycore_score: {scored.spotifycore_score}")
        print(f"  components:")
        for k, v in scored.components.items():
            print(f"    {k}: {v}")

        assert isinstance(scored, ScoredTrack), "Scored is not a ScoredTrack instance"
        assert 0 <= scored.engagement_score <= 100
        assert 0 <= scored.spotifycore_score <= 100

        print("\nAll assertions passed.")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_mir_analyzer()
