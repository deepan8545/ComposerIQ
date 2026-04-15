"""
Train XGBoost scoring calibrator from real audio analysis.

Reads benchmark_manifest.json, runs librosa feature extraction + MIR
analysis on each track's actual audio file, computes the same 6 signals
used at inference time, and trains an XGBRegressor.

This eliminates the train/inference feature mismatch — the model trains
on the exact same librosa-derived signals it will see in production.

Changes in v3.0:
  - Added 'hook' signal to SIGNAL_NAMES (was missing before)
  - Improved quality_label: includes hook_percentile + skip_risk_pct
  - 5-fold cross-validation for honest R² estimate
  - Full-dataset retrain for the deployed model

Usage:
    python train_calibrator.py
"""
import json
import sys
import time
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score

from pipelines.audio_analysis import AudioAnalyzer
from data.mir_analyzer import MIRAnalyzer
from pipelines.scoring_engine import (
    _hook_signal,
    _skip_risk_signal,
    _energy_signal,
    _danceability_signal,
    _valence_signal,
    _tempo_signal,
)

MANIFEST_PATH = Path(__file__).parent / "benchmark_manifest.json"
CLUSTER_STATS_PATH = Path(__file__).parent / "cluster_stats.json"
MODELS_DIR = Path(__file__).parent / "models"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"

# All 6 signals — MUST match scoring_engine.py at inference time
SIGNAL_NAMES = ["hook", "skip_inv", "energy", "danceability", "valence", "tempo"]


# ---------------------------------------------------------------------------
# Step 1: Load manifest + cluster stats
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    """Flatten benchmark_manifest.json into a list of tracks with cluster info."""
    with open(MANIFEST_PATH) as f:
        data = json.load(f)

    tracks = []
    for cluster_key, cluster in data.items():
        cluster_id = int(cluster_key.replace("cluster_", ""))
        genre = cluster.get("genre", "unknown")
        for t in cluster.get("tracks", []):
            t["cluster_id"] = cluster_id
            t["genre"] = genre.lower()
            tracks.append(t)

    return tracks


def load_cluster_stats() -> dict:
    """Load cluster stats for MIR analysis."""
    if CLUSTER_STATS_PATH.exists():
        with open(CLUSTER_STATS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Step 2: Analyze each track with real librosa features
# ---------------------------------------------------------------------------

def analyze_tracks(tracks: list[dict], cluster_stats: dict) -> tuple[list[dict], list[dict]]:
    """
    Run AudioAnalyzer + MIRAnalyzer on each track's audio file.
    Returns (processed, skipped) lists.
    """
    audio_analyzer = AudioAnalyzer()
    mir_analyzer = MIRAnalyzer()

    processed = []
    skipped = []

    for i, t in enumerate(tracks):
        audio_path = t.get("audio_path", "")
        name = t.get("name", "Unknown")
        artist = t.get("artist", "Unknown")
        label = f"{name} by {artist}"

        # Validate audio path
        if not audio_path:
            skipped.append({"track": label, "reason": "no audio_path"})
            continue

        full_path = Path(__file__).parent / audio_path
        if not full_path.exists():
            skipped.append({"track": label, "reason": f"file not found: {audio_path}"})
            continue

        try:
            # Layer 1: librosa feature extraction
            features = audio_analyzer.analyze(str(full_path))

            # Layer 2: MIR percentiles
            cid = str(t.get("cluster_id", 0))
            stats = cluster_stats.get(cid, {
                "skip_risk_mean": 0.13,
                "skip_risk_std": 0.10,
            })
            genre = t.get("genre", "pop")
            mir = mir_analyzer.analyze(features, stats, genre=genre)

            # Compute all 6 signals using the exact same functions as scoring_engine.py.
            # This eliminates any train/inference signal mismatch.
            signals = {
                "hook":         _hook_signal(mir.hook_time),
                "skip_inv":     _skip_risk_signal(mir.skip_risk_percentile),
                "energy":       _energy_signal(features.get("energy", 0.5)),
                "danceability": _danceability_signal(features.get("danceability", 0.5)),
                "valence":      _valence_signal(features.get("valence", 0.5)),
                "tempo":        _tempo_signal(features.get("tempo", 120.0)),
            }

            # -------------------------------------------------------------------
            # Quality label v3.0 — multi-signal engagement proxy
            #
            # Weights reflect Spotify algorithmic promotion drivers:
            #   popularity      (Spotify's actual engagement measurement) : 50%
            #   skip_inv_norm   (lower skip risk = higher engagement)     : 20%
            #   hook_pct        (earlier hook = better algorithmic rank)  : 15%
            #   energy_norm     (high energy = better engagement)         : 10%
            #   dance_norm      (danceability = playlist fit)             : 5%
            #
            # Key fix vs v2.0: hook signal now IN the label, making the XGBoost
            # model consistent with the formula it blends with at inference time.
            # -------------------------------------------------------------------
            popularity    = float(t.get("popularity", 0))
            hook_pct      = float(mir.hook_percentile)                    # 0-100
            skip_inv_norm = 100.0 - float(mir.skip_risk_percentile)      # higher = better
            energy_norm   = float(np.clip(features.get("energy", 0.5) * 100, 0, 100))
            dance_norm    = float(np.clip(features.get("danceability", 0.5) * 100, 0, 100))

            quality_label = (
                popularity    * 0.50 +
                skip_inv_norm * 0.20 +
                hook_pct      * 0.15 +
                energy_norm   * 0.10 +
                dance_norm    * 0.05
            )

            processed.append({
                "name": name,
                "artist": artist,
                "track_id": t.get("track_id", ""),
                "cluster_id": t.get("cluster_id", 0),
                "popularity": popularity,
                "signals": signals,
                "quality_label": round(quality_label, 2),
                "features": {
                    "energy": features.get("energy", 0),
                    "danceability": features.get("danceability", 0),
                    "valence": features.get("valence", 0),
                    "tempo": features.get("tempo", 0),
                    "hook_time": mir.hook_time,
                    "hook_percentile": mir.hook_percentile,
                    "skip_risk_percentile": mir.skip_risk_percentile,
                },
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(tracks)}] Processed {label}")

        except Exception as e:
            skipped.append({"track": label, "reason": str(e)})

    return processed, skipped


# ---------------------------------------------------------------------------
# Step 3: Train XGBoost with 5-fold CV + full-dataset retrain
# ---------------------------------------------------------------------------

def train_model(processed: list[dict]):
    """
    Train XGBRegressor on the computed signals.

    Process:
      1. 5-fold cross-validation for honest R² estimate
      2. Retrain on full dataset for the production model
      3. Hold-out 20% for final spot-check metrics
    """
    X = np.array([[rec["signals"][k] for k in SIGNAL_NAMES] for rec in processed])
    y = np.array([rec["quality_label"] for rec in processed])

    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Label range: {y.min():.1f} - {y.max():.1f} "
          f"(mean={y.mean():.1f}, std={y.std():.1f})")

    # Signal statistics
    print(f"\n  Signal statistics:")
    for i, name in enumerate(SIGNAL_NAMES):
        col = X[:, i]
        print(f"    {name:>15}: mean={col.mean():.1f}, std={col.std():.1f}, "
              f"range=[{col.min():.1f}, {col.max():.1f}]")

    # ---- 5-fold cross-validation ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes, fold_r2s = [], []

    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    print(f"\n  5-Fold Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = XGBRegressor(**xgb_params)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2  = r2_score(y_val, preds)
        fold_maes.append(mae)
        fold_r2s.append(r2)
        print(f"    Fold {fold}: MAE={mae:.2f}  R\u00b2={r2:.3f}")

    cv_mae_mean = float(np.mean(fold_maes))
    cv_mae_std  = float(np.std(fold_maes))
    cv_r2_mean  = float(np.mean(fold_r2s))
    cv_r2_std   = float(np.std(fold_r2s))
    print(f"  CV: MAE={cv_mae_mean:.2f}\u00b1{cv_mae_std:.2f}  "
          f"R\u00b2={cv_r2_mean:.3f}\u00b1{cv_r2_std:.3f}")

    # ---- Retrain on full dataset for deployment ----
    print(f"\n  Retraining on full {len(X)} samples for production model...")
    model = XGBRegressor(**xgb_params)
    model.fit(X, y)

    # Feature importances
    importances = dict(zip(SIGNAL_NAMES, model.feature_importances_.tolist()))
    print(f"\n  Feature importances (ranked):")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {name:>15}: {imp:.3f}  {bar}")

    # Hold-out metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    test_model = XGBRegressor(**xgb_params)
    test_model.fit(X_train, y_train)
    test_pred  = test_model.predict(X_test)
    test_mae   = mean_absolute_error(y_test, test_pred)
    test_r2    = r2_score(y_test, test_pred)

    return model, importances, {
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std":  cv_mae_std,
        "cv_r2_mean":  cv_r2_mean,
        "cv_r2_std":   cv_r2_std,
        "test_mae":    test_mae,
        "test_r2":     test_r2,
        "n_train":     len(X),
        "n_test":      len(X_test),
    }


# ---------------------------------------------------------------------------
# Step 4: Save model
# ---------------------------------------------------------------------------

def save_model(model, importances, processed, metrics):
    """Save calibrator to models/calibrator.joblib."""
    MODELS_DIR.mkdir(exist_ok=True)

    model_bundle = {
        "xgb": model,
        "signal_names": SIGNAL_NAMES,
        "feature_importances": importances,
        "n_training_samples": metrics["n_train"],
        "version": "3.0",
        "scoring_mode": "xgboost",
        "cv_r2_mean": metrics["cv_r2_mean"],
        "cv_r2_std":  metrics["cv_r2_std"],
    }

    joblib.dump(model_bundle, CALIBRATOR_PATH)
    print(f"\n  Model saved to {CALIBRATOR_PATH}")
    print(f"  Version: 3.0 (all 6 signals incl. hook, 5-fold CV validated)")

    # Save training data for reproducibility
    training_data_path = MODELS_DIR / "training_data.json"
    with open(training_data_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"  Training data saved to {training_data_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  ComposerIQ - XGBoost Calibrator Training (v3.0)")
    print("  Signals: hook + skip_inv + energy + dance + valence + tempo")
    print("  Label:   pop(50%) + skip_inv(20%) + hook_pct(15%) + energy(10%) + dance(5%)")
    print("=" * 64)

    # Load data
    print("\n[Step 1] Loading manifest and cluster stats...")
    tracks = load_manifest()
    cluster_stats = load_cluster_stats()
    print(f"  Manifest: {len(tracks)} tracks across {len(cluster_stats)} clusters")

    # Analyze tracks
    print(f"\n[Step 2] Analyzing {len(tracks)} tracks with librosa + MIR...")
    print(f"  (This takes ~30-60s for 176 tracks)\n")
    start = time.time()
    processed, skipped = analyze_tracks(tracks, cluster_stats)
    elapsed = time.time() - start
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Processed: {len(processed)} tracks")
    print(f"  Skipped:   {len(skipped)} tracks")

    if skipped:
        print(f"\n  Skip reasons:")
        reasons: dict[str, int] = {}
        for s in skipped:
            r = s["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:>3}x {reason[:80]}")

    if len(processed) < 10:
        print(f"\n[FAILED] Only {len(processed)} tracks processed — need at least 10.")
        sys.exit(1)

    # Train with CV
    print(f"\n[Step 3] Training XGBoost with 5-fold cross-validation...")
    model, importances, metrics = train_model(processed)

    # Save
    print(f"\n[Step 4] Saving model...")
    save_model(model, importances, processed, metrics)

    # Final report
    print(f"\n{'=' * 64}")
    print(f"  TRAINING COMPLETE")
    print(f"  Model:        {CALIBRATOR_PATH}")
    print(f"  Tracks:       {len(processed)} processed, {len(skipped)} skipped")
    print(f"  CV MAE:       {metrics['cv_mae_mean']:.2f} \u00b1 {metrics['cv_mae_std']:.2f}")
    print(f"  CV R\u00b2:        {metrics['cv_r2_mean']:.3f} \u00b1 {metrics['cv_r2_std']:.3f}")
    print(f"  Hold-out MAE: {metrics['test_mae']:.2f}")
    print(f"  Hold-out R\u00b2:  {metrics['test_r2']:.3f}")
    print(f"  Signals:      {', '.join(SIGNAL_NAMES)}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
