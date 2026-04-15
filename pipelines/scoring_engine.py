"""
Layer 3: Scoring Engine
-----------------------
Takes raw features (Layer 1) and MIR percentiles (Layer 2),
outputs calibrated engagement and spotifycore scores (0-100).

This is the ONLY place where final scores are computed.
Claude (Layer 4) can only adjust within a narrow band.

Scoring modes:
  - If models/calibrator.joblib exists: blends GBR model prediction
    with weighted formula + excellence boosts.
  - Otherwise: uses weighted formula + excellence boosts alone.
"""
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
import numpy as np
import joblib

from data.mir_analyzer import MIRResult

# ---------------------------------------------------------------------------
# Learned model loading
# ---------------------------------------------------------------------------
_CALIBRATOR_PATH = Path(__file__).parent.parent / "models" / "calibrator.joblib"
_learned_model = None

def _load_learned_model():
    """Load the trained calibrator if available."""
    global _learned_model
    if _CALIBRATOR_PATH.exists():
        try:
            _learned_model = joblib.load(_CALIBRATOR_PATH)
            importances = _learned_model.get("feature_importances", {})
            n_samples = _learned_model.get("n_training_samples", 0)
            version = _learned_model.get("version", "1.0")
            mode = _learned_model.get("scoring_mode", "unknown")
            print(f"[Scoring] Loaded calibrator v{version} ({mode}, {n_samples} samples)")
            print(f"[Scoring] Importances: " + ", ".join(
                f"{k}={v:.3f}" for k, v in importances.items()
            ))
        except Exception as e:
            print(f"[Scoring] Failed to load calibrator: {e}")
            _learned_model = None
    else:
        print("[Scoring] No learned calibrator found — using formula only")

_load_learned_model()


@dataclass
class ScoredTrack:
    """Final calibrated scores — this is what the frontend sees."""
    engagement_score: int       # 0-100
    spotifycore_score: int      # 0-100
    components: Dict[str, float]
    hook_time: Optional[float]
    hook_percentile: float
    skip_risk_percentile: float
    factor_breakdown: Dict[str, float]


# ---------------------------------------------------------------------------
# Feature-to-signal transforms (all return 0-100)
# ---------------------------------------------------------------------------

def _hook_signal(hook_time: Optional[float]) -> float:
    """Hook timing signal. Early hooks score higher."""
    ht = hook_time if hook_time is not None else 30.0
    if ht <= 5:    return 100.0
    elif ht <= 10: return 90.0 - (ht - 5) * 2
    elif ht <= 15: return 80.0 - (ht - 10) * 2
    elif ht <= 20: return 70.0 - (ht - 15) * 2
    elif ht <= 25: return 60.0 - (ht - 20) * 6
    else:          return max(0, 30.0 - (ht - 25) * 6)


def _skip_risk_signal(skip_risk_pct: float) -> float:
    """Inverse skip risk (higher = BETTER = lower risk)."""
    return max(0.0, 100.0 - skip_risk_pct)


def _energy_signal(energy: float) -> float:
    """Energy signal. Linear with soft ceiling."""
    return float(np.clip(energy * 130, 0, 100))


def _danceability_signal(danceability: float) -> float:
    """Danceability signal. Convex — rewards high values more."""
    if danceability > 0.95:   return 100.0
    elif danceability > 0.90: return 95.0
    elif danceability > 0.80: return 85.0
    elif danceability > 0.70: return 72.0
    elif danceability > 0.50: return 55.0
    else:                     return danceability * 80


def _valence_signal(valence: float) -> float:
    """Valence signal. Floor at 40 — dark tracks can be excellent."""
    return float(np.clip(40 + valence * 65, 40, 100))


def _tempo_signal(tempo: float) -> float:
    """Tempo signal. Wide sweet spot: 80-145 BPM."""
    if 80 <= tempo <= 145:    return 100.0
    elif 70 <= tempo < 80 or 145 < tempo <= 160: return 80.0
    elif 60 <= tempo < 70 or 160 < tempo <= 175: return 60.0
    else:                     return 40.0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_ENGAGEMENT_WEIGHTS = {
    'hook':         0.18,
    'skip_inv':     0.18,
    'energy':       0.22,
    'danceability': 0.22,
    'valence':      0.10,
    'tempo':        0.10,
}

_SPOTIFYCORE_WEIGHTS = {
    'hook':         0.15,
    'skip_inv':     0.15,
    'energy':       0.20,
    'danceability': 0.25,
    'valence':      0.12,
    'tempo':        0.13,
}


# ---------------------------------------------------------------------------
# Human-readable fix hints per signal (used in counterfactual output)
# ---------------------------------------------------------------------------

_SIGNAL_FIX_HINTS = {
    'hook':         "Move the hook (chorus/drop) to before {target_sec:.0f}s — currently at {current_sec:.0f}s",
    'skip_inv':     "Strengthen the intro energy to reduce skip risk (add a hook cue or build within 15s)",
    'energy':       "Increase overall loudness/RMS — target -6 to -10 LUFS integrated loudness",
    'danceability': "Tighten beat regularity — quantise drums or use a steadier groove template",
    'valence':      "Add brighter chord voicings or shift from minor to mixed-mode harmony",
    'tempo':        "Adjust BPM to the 80-145 BPM sweet spot for streaming placement",
}


class ScoringEngine:
    """
    Scoring engine with optional ML model blending.

    Architecture:
      1. Compute 6 signals (0-100 each)
      2. Weighted average → base score
      3. Non-linear excellence boosts for standout tracks
      4. If learned model exists: blend model prediction with formula
         (model captures interaction effects formula can't)
    """

    def __init__(self):
        self._eng_weights = dict(_ENGAGEMENT_WEIGHTS)
        self._spot_weights = dict(_SPOTIFYCORE_WEIGHTS)
        self._xgb = None
        self._model_blend = 0.0  # 0 = formula only, >0 = blend with model

        if _learned_model is not None and "xgb" in _learned_model:
            self._xgb = _learned_model["xgb"]
            self._model_signal_names = _learned_model.get("signal_names", [])
            # Blend weight: how much to trust the model vs formula
            # Start low (0.3) — increase as model improves with more data
            self._model_blend = 0.3
            self._mode = "hybrid"
        else:
            self._mode = "formula"

    def _compute_excellence_boost(self, signals: dict, features: dict) -> float:
        """
        Non-linear boost for tracks with multiple strong signals.
        A track with 4+ strong signals is clearly well-produced —
        the weighted average alone can't push it past ~85.
        """
        strong = sum(1 for v in signals.values() if v >= 75)
        exceptional = sum(1 for v in signals.values() if v >= 90)

        boost = 0.0

        # Strong signal count bonus
        if strong >= 5:
            boost += 8.0
        elif strong >= 4:
            boost += 5.0
        elif strong >= 3:
            boost += 2.0

        # Exceptional signal bonus
        if exceptional >= 3:
            boost += 6.0
        elif exceptional >= 2:
            boost += 3.0
        elif exceptional >= 1:
            boost += 1.0

        # Hook confidence bonus
        hook_conf = features.get('hook_confidence', 0.5)
        if hook_conf >= 0.9:
            boost += 3.0
        elif hook_conf >= 0.7:
            boost += 1.5

        return boost

    def compute_counterfactuals(
        self,
        signals: dict,
        features: dict,
        poor_threshold: float = 65.0,
        target_signal: float = 78.0,
    ) -> list:
        """
        For each signal below `poor_threshold`, compute the engagement score delta
        that would result if that signal were raised to `target_signal`.

        Returns a list of dicts sorted by score_delta descending, e.g.::

            [
                {
                    "signal":      "hook",
                    "current":     42.0,
                    "target":      78.0,
                    "score_delta": +8.2,
                    "fix_hint":    "Move the hook to before 15s ..."
                },
                ...
            ]
        """
        # Baseline engagement (formula only, no model blending for counterfactuals)
        baseline = sum(signals[k] * self._eng_weights[k] for k in signals)
        baseline += self._compute_excellence_boost(signals, features)

        results = []
        for sig_name in self._eng_weights:
            current_val = signals.get(sig_name, 0.0)
            if current_val >= poor_threshold:
                continue  # signal already strong — skip

            # Compute hypothetical score with this one signal raised
            hypo_signals = dict(signals)
            hypo_signals[sig_name] = target_signal
            hypo_eng = sum(hypo_signals[k] * self._eng_weights[k] for k in hypo_signals)
            hypo_eng += self._compute_excellence_boost(hypo_signals, features)
            delta = hypo_eng - baseline

            # Build a readable fix hint
            hint_template = _SIGNAL_FIX_HINTS.get(sig_name, f"Improve {sig_name}")
            if sig_name == 'hook':
                # Convert hook signal to approximate seconds for the hint
                # Invert _hook_signal: score 78 ≈ hook_time ~13s; score 42 ≈ ~22s
                def _approx_hook_sec(score: float) -> float:
                    if score >= 100: return 5.0
                    if score >= 90:  return 10.0
                    if score >= 80:  return 15.0
                    if score >= 60:  return 20.0
                    return 25.0
                hint = hint_template.format(
                    target_sec=_approx_hook_sec(target_signal),
                    current_sec=_approx_hook_sec(current_val),
                )
            else:
                hint = hint_template

            results.append({
                "signal":      sig_name,
                "current":     round(current_val, 1),
                "target":      round(target_signal, 1),
                "score_delta": round(delta, 1),
                "fix_hint":    hint,
            })

        results.sort(key=lambda x: -x["score_delta"])
        return results

    def score(self, features: dict, mir: MIRResult) -> ScoredTrack:
        # Compute 6 signals (all 0-100)
        signals = {
            'hook':         _hook_signal(mir.hook_time),
            'skip_inv':     _skip_risk_signal(mir.skip_risk_percentile),
            'energy':       _energy_signal(features.get('energy', 0.5)),
            'danceability': _danceability_signal(features.get('danceability', 0.5)),
            'valence':      _valence_signal(features.get('valence', 0.5)),
            'tempo':        _tempo_signal(features.get('tempo', 120.0)),
        }

        # --- Formula-based scoring ---
        eng_formula = sum(signals[k] * self._eng_weights[k] for k in signals)
        spot_formula = sum(signals[k] * self._spot_weights[k] for k in signals)

        # Excellence boost
        boost = self._compute_excellence_boost(signals, features)

        eng_formula += boost
        spot_formula += boost

        # --- Model blending (if available) ---
        if self._xgb is not None and self._model_blend > 0:
            try:
                X = np.array([[signals[k] for k in self._model_signal_names]])
                model_pred = float(self._xgb.predict(X)[0])
                # Blend: formula dominates, model adds nuance
                b = self._model_blend
                engagement_raw = (1 - b) * eng_formula + b * model_pred
                spotifycore_raw = (1 - b) * spot_formula + b * model_pred
            except Exception:
                # Model failed — fall back to formula
                engagement_raw = eng_formula
                spotifycore_raw = spot_formula
        else:
            engagement_raw = eng_formula
            spotifycore_raw = spot_formula

        # Clamp to 0-100
        engagement = int(np.clip(round(engagement_raw), 0, 100))
        spotifycore = int(np.clip(round(spotifycore_raw), 0, 100))

        print(f"[Scoring] Signals: " + ", ".join(
            f"{k}={v:.0f}" for k, v in signals.items()
        ))
        print(f"[Scoring] base_eng={eng_formula:.0f}, base_spot={spot_formula:.0f}, "
              f"boost=+{boost:.0f}, mode={self._mode}")
        print(f"[Scoring] engagement={engagement}, spotifycore={spotifycore}")

        return ScoredTrack(
            engagement_score=engagement,
            spotifycore_score=spotifycore,
            components={k: round(v, 1) for k, v in signals.items()},
            hook_time=mir.hook_time,
            hook_percentile=mir.hook_percentile,
            skip_risk_percentile=mir.skip_risk_percentile,
            factor_breakdown=mir.factor_breakdown,
        )
