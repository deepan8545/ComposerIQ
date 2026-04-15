"""
Layer 2: MIR Percentiles
-------------------------
Takes raw features from Layer 1 and computes genre-conditional
percentile ranks. Returns ONLY percentiles and factor breakdowns —
no composite scores, no engagement numbers.
All final scoring is done in scoring_engine.py (Layer 3).
"""
from dataclasses import dataclass
from typing import Optional, Dict
import librosa
import numpy as np


@dataclass
class MIRResult:
    """Pure MIR output — percentiles and measurements, no composite scores."""
    hook_time: Optional[float]
    hook_confidence: float
    # Percentile ranks (0-100, higher = better/earlier/lower-risk)
    hook_percentile: float          # vs genre top-10% tier
    skip_risk_percentile: float     # vs cluster distribution
    # Raw factor measurements (0-1 each)
    factor_breakdown: Dict[str, float]
    raw_risk: float                 # weighted composite of factors (0-1)
    cluster_id: int
    genre_tier: str                 # which tier was used for comparison


# ---------------------------------------------------------------------------
# Genre-stratified reference distributions
# Top-10% hits legitimately have later hooks (earned build-ups).
# These are the reference classes for P(T_hook <= t | genre, tier).
# ---------------------------------------------------------------------------
STRATIFIED_CLUSTERS = {
    'pop': {
        'top_10': {'hook_mean': 13.0, 'hook_std': 5.0, 'skip_mean': 0.10, 'skip_std': 0.08},
        'median': {'hook_mean': 8.0, 'hook_std': 3.0, 'skip_mean': 0.15, 'skip_std': 0.10},
    },
    'hip-hop': {
        'top_10': {'hook_mean': 10.0, 'hook_std': 4.0, 'skip_mean': 0.08, 'skip_std': 0.06},
        'median': {'hook_mean': 7.0, 'hook_std': 2.5, 'skip_mean': 0.12, 'skip_std': 0.09},
    },
    'r&b': {
        'top_10': {'hook_mean': 14.0, 'hook_std': 5.5, 'skip_mean': 0.12, 'skip_std': 0.09},
        'median': {'hook_mean': 9.0, 'hook_std': 3.5, 'skip_mean': 0.16, 'skip_std': 0.11},
    },
    'electronic': {
        'top_10': {'hook_mean': 16.0, 'hook_std': 6.0, 'skip_mean': 0.11, 'skip_std': 0.08},
        'median': {'hook_mean': 10.0, 'hook_std': 4.0, 'skip_mean': 0.14, 'skip_std': 0.10},
    },
    'indie pop': {
        'top_10': {'hook_mean': 15.0, 'hook_std': 5.5, 'skip_mean': 0.13, 'skip_std': 0.09},
        'median': {'hook_mean': 9.5, 'hook_std': 3.5, 'skip_mean': 0.17, 'skip_std': 0.11},
    },
    'ambient': {
        'top_10': {'hook_mean': 25.0, 'hook_std': 10.0, 'skip_mean': 0.20, 'skip_std': 0.12},
        'median': {'hook_mean': 18.0, 'hook_std': 7.0, 'skip_mean': 0.25, 'skip_std': 0.14},
    },
}
_DEFAULT_TIER = {'hook_mean': 12.0, 'hook_std': 5.0, 'skip_mean': 0.12, 'skip_std': 0.09}


class MIRAnalyzer:
    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop = hop_length

    def load(self, path: str):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        return y

    # ------------------------------------------------------------------
    # Hook detection (onset + chroma + beat alignment)
    # ------------------------------------------------------------------
    def _detect_hook(self, y, analysis_window_sec=90):
        hop = self.hop
        sr = self.sr

        max_frames = librosa.time_to_frames(
            min(analysis_window_sec, librosa.get_duration(y=y, sr=sr) * 0.6),
            sr=sr, hop_length=hop
        )

        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop, aggregate=np.median
        )
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        chroma_flux = np.concatenate([
            [0], np.linalg.norm(np.diff(chroma, axis=1), axis=0)
        ])
        _, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop, trim=False
        )
        beat_mask = np.zeros(len(onset_env), dtype=bool)
        beat_mask[beat_frames[beat_frames < len(onset_env)]] = True

        window = onset_env[:max_frames]
        c_window = chroma_flux[:max_frames]

        onset_thresh = np.percentile(window, 75)
        chroma_thresh = np.percentile(c_window, 75)
        energy_gate = np.percentile(window, 60)

        candidates = (
            (onset_env[:max_frames] > onset_thresh) &
            (chroma_flux[:max_frames] > chroma_thresh) &
            (onset_env[:max_frames] > energy_gate) &
            beat_mask[:max_frames]
        )

        peak_energy = np.max(window) if len(window) else 1.0
        silence_mask = onset_env[:max_frames] < (peak_energy * 0.30)
        skip_frames = librosa.time_to_frames(2, sr=sr, hop_length=hop)
        candidates[:skip_frames] = False
        suppress_range = min(librosa.time_to_frames(8, sr=sr, hop_length=hop), max_frames)
        candidates[skip_frames:suppress_range] &= ~silence_mask[skip_frames:suppress_range]

        frames = np.where(candidates)[0]
        if not len(frames):
            candidates_relaxed = (
                (onset_env[:max_frames] > onset_thresh) &
                (chroma_flux[:max_frames] > chroma_thresh) &
                beat_mask[:max_frames]
            )
            candidates_relaxed[:skip_frames] = False
            candidates_relaxed[skip_frames:suppress_range] &= ~silence_mask[skip_frames:suppress_range]
            frames = np.where(candidates_relaxed)[0]

        if not len(frames):
            return None, 0.0

        t = float(librosa.frames_to_time(frames[0], sr=sr, hop_length=hop))
        conf = float(np.clip(
            (onset_env[frames[0]] / (onset_thresh + 1e-8) +
             chroma_flux[frames[0]] / (chroma_thresh + 1e-8)) / 6,
            0, 1
        ))
        return t, conf

    # ------------------------------------------------------------------
    # Sigmoid percentile: softer than raw CDF, avoids extreme scores
    # ------------------------------------------------------------------
    @staticmethod
    def _z_to_pct(val, mean, std, invert=False):
        z = (mean - val if invert else val - mean) / (std + 1e-8)
        return float(np.clip((1 + np.tanh(z * 0.5)) / 2 * 100, 0, 100))

    # ------------------------------------------------------------------
    # Compute risk factors from raw feature measurements
    # ------------------------------------------------------------------
    @staticmethod
    def compute_factors(hook_time: Optional[float],
                        energy_slope: float,
                        spectral_variance: float,
                        onset_density: float) -> dict:
        """Convert raw measurements into 0-1 risk factors."""
        # Non-linear hook delay: under 20s is low risk
        ht = hook_time if hook_time is not None else 30.0
        if ht <= 20:
            hook_delay = (ht / 20) ** 2 * 0.3
        else:
            hook_delay = 0.3 + (ht - 20) / 10 * 0.7

        return {
            'energy_slope':   float(np.clip(-energy_slope * 1e4, 0, 1)),
            'hook_delay':     float(np.clip(hook_delay, 0, 1)),
            'spec_variance':  float(np.clip(1 - spectral_variance * 5, 0, 1)),
            'onset_density':  float(np.clip(1 - onset_density * 3, 0, 1)),
        }

    # ------------------------------------------------------------------
    # Main analyze: features in → percentiles out
    # ------------------------------------------------------------------
    def analyze(self, features: dict, cluster_stats: dict,
                genre: str = "pop") -> MIRResult:
        """
        Takes raw features (from Layer 1) and cluster stats,
        returns genre-conditional percentile ranks.

        P2 update: prefers hook_time_mert (MERT frame-level estimate) when
        its confidence exceeds 0.6 — falls back to librosa estimate otherwise.
        """
        # --- Hook time selection (P2: MERT preferred when confident) ---
        hook_time_mert  = features.get('hook_time_mert')
        hook_conf_mert  = features.get('hook_confidence_mert', 0.0)
        hook_time_lib   = features.get('hook_time')

        if hook_time_mert is not None and hook_conf_mert >= 0.6:
            hook_time = hook_time_mert
            hook_conf = hook_conf_mert
            hook_source = f"MERT (conf={hook_conf_mert:.2f})"
        else:
            hook_time = hook_time_lib
            hook_conf = features.get('hook_confidence', 0.0)
            hook_source = "librosa (MERT conf too low or unavailable)"

        print(f"[MIR] Hook source: {hook_source} → hook_time={hook_time}")

        # Compute risk factors from raw measurements
        factors = self.compute_factors(
            hook_time=hook_time,
            energy_slope=features.get('energy_slope', 0.0),
            spectral_variance=features.get('spectral_variance', 0.2),
            onset_density=features.get('onset_density', 0.3),
        )

        # Weighted composite raw risk (0-1 scale)
        raw_risk = (
            factors['energy_slope']  * 0.30 +
            factors['hook_delay']    * 0.35 +
            factors['spec_variance'] * 0.20 +
            factors['onset_density'] * 0.15
        )

        # Genre-stratified hook percentile: P(T_hook <= t | genre, top_10)
        tier = STRATIFIED_CLUSTERS.get(genre, {}).get('top_10', _DEFAULT_TIER)
        hook_percentile = self._z_to_pct(
            hook_time if hook_time is not None else 30.0,
            tier['hook_mean'],
            tier['hook_std'],
            invert=True   # earlier = higher percentile
        )

        # Skip risk percentile: calibrated against cluster distribution
        skip_risk_pct = self._z_to_pct(
            raw_risk,
            cluster_stats.get('skip_risk_mean', tier['skip_mean']),
            cluster_stats.get('skip_risk_std', tier['skip_std']),
            invert=False  # higher raw_risk = higher skip risk percentile
        )

        return MIRResult(
            hook_time=round(hook_time, 2) if hook_time is not None else None,
            hook_confidence=round(hook_conf, 3),
            hook_percentile=round(hook_percentile, 1),
            skip_risk_percentile=round(skip_risk_pct, 1),
            factor_breakdown={k: round(v, 4) for k, v in factors.items()},
            raw_risk=round(raw_risk, 4),
            cluster_id=cluster_stats.get('cluster_id', -1),
            genre_tier=f"{genre}/top_10",
        )
