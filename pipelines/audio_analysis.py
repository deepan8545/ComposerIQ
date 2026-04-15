"""
Layer 1: Pure Feature Extraction
---------------------------------
Extracts raw audio measurements from a waveform.
Returns ONLY measurements — no scores, no percentiles, no risk labels.
All scoring is done downstream in scoring_engine.py.

New in this version:
  - chord_change_rate   : # chroma column transitions per second (harmonic activity)
  - harmonic_complexity : entropy of mean chroma vector (flat = more complex)
  - tonal_stability     : 1 - std of dominant pitch class over time (1 = stable key)
"""
import json
import librosa
import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import TypedDict, List, Optional
import traceback


class AudioFeatures(TypedDict):
    tempo: float
    duration: float
    key: str
    hook_time: Optional[float]
    hook_confidence: float
    hook_time_mert: Optional[float]       # MERT frame-level hook estimate (P2)
    hook_confidence_mert: float           # MERT hook confidence (0-1)
    energy_curve: List[float]
    danceability: float
    energy: float
    valence: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    loudness: float
    # Raw risk factors (measurements, not scores)
    energy_slope: float
    spectral_variance: float
    onset_density: float
    # Harmonic complexity features (P3)
    chord_change_rate: float    # chroma transitions per second (higher = more chord changes)
    harmonic_complexity: float  # entropy of chroma distribution (higher = richer harmony)
    tonal_stability: float      # 1 - normalised std of dominant pitch class (higher = more stable)
    # Structural segmentation (P5)
    segments: List[dict]        # [{"label": str, "start": float, "end": float}]


def to_float(val):
    try:
        if hasattr(val, '__len__'):
            return float(np.mean(val))
        return float(val)
    except Exception:
        return 0.0


class AudioAnalyzer:
    KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def analyze(self, audio_path: str) -> AudioFeatures:
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"[Features] Audio loaded: duration={duration:.1f}s, samples={len(y)}")

            # ── Energy curve (first 30s in 5s windows = 6 values) ──
            energy_curve = []
            segment_len = sr * 5
            for i in range(6):
                start = i * segment_len
                end = start + segment_len
                segment = y[start:end]
                if len(segment) > 0:
                    val = float(np.sqrt(np.mean(segment ** 2)))
                    energy_curve.append(round(val, 6))
                else:
                    energy_curve.append(0.0)

            # ── Hook detection (librosa single source of truth + MERT fusion) ──
            from data.mir_analyzer import MIRAnalyzer
            _mir = MIRAnalyzer(sr=sr, hop_length=512)
            hook_time, hook_conf = _mir._detect_hook(y)
            print(f"[Features] Hook (librosa): {hook_time:.1f}s (conf={hook_conf:.2f})" if hook_time else "[Features] Hook (librosa): not detected")

            # MERT frame-level hook detection (P2)
            try:
                from mert_embedder import embedder as _mert_embedder
                from pipelines.mert_hook_detector import detect_hook_mert
                hook_time_mert, hook_conf_mert = detect_hook_mert(
                    audio_path=audio_path,
                    embedder=_mert_embedder,
                    librosa_hook_time=hook_time,
                )
                print(
                    f"[Features] Hook (MERT): {hook_time_mert:.1f}s (conf={hook_conf_mert:.2f})"
                    if hook_time_mert else "[Features] Hook (MERT): not detected"
                )
            except Exception as e:
                print(f"[Features] MERT hook detection failed: {e}")
                hook_time_mert, hook_conf_mert = None, 0.0

            # ── Raw risk factors (measurements only — scoring is in scoring_engine) ──
            hop = 512
            y_90 = y[:sr * 90]
            rms = librosa.feature.rms(y=y_90, hop_length=hop)[0]
            frames_10s = librosa.time_to_frames(10, sr=sr, hop_length=hop)

            # Energy slope in first 10s (negative = energy dropping = bad)
            intro_rms = rms[:frames_10s]
            energy_slope = float(np.polyfit(np.arange(len(intro_rms)), intro_rms, deg=1)[0])

            # Spectral variance in first 10s (low = boring intro)
            centroid = librosa.feature.spectral_centroid(y=y_90, sr=sr)[0]
            spectral_variance = float(
                np.std(centroid[:frames_10s]) / (np.mean(centroid[:frames_10s]) + 1e-8)
            )

            # Onset density (low = sparse intro)
            onset_env = librosa.onset.onset_strength(y=y_90, sr=sr, hop_length=hop)
            onset_density = float(np.mean(onset_env > np.percentile(onset_env, 60)))

            # ── Tempo + key ──
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            rms_full = librosa.feature.rms(y=y, hop_length=512)[0]
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key = self.KEY_NAMES[int(np.argmax(np.mean(chroma, axis=1))) % 12]

            # ── Spectral features ──
            centroid_full = librosa.feature.spectral_centroid(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mean_centroid = float(np.mean(centroid_full))

            # ── ENERGY (RMS normalized to 0-1) ──
            rms_mean = float(np.mean(rms_full))
            energy_val = float(np.clip(rms_mean / 0.3, 0.0, 1.0))

            # ── LOUDNESS (70th percentile dB, ref=1.0 for realistic negative values) ──
            rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y), ref=1.0)
            loudness = float(np.percentile(rms_db, 70))

            # ── SPEECHINESS (spectral contrast in voice bands 300-3000 Hz) ──
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            voice_contrast = float(np.mean(spectral_contrast[1:4]))
            speechiness = float(np.clip(voice_contrast / 30.0, 0.03, 0.95))

            # ── INSTRUMENTALNESS (3-signal vocal detection) ──
            mfcc_var = float(np.mean(np.std(mfcc[2:7], axis=1)))
            vocal_signal_1 = float(np.clip(mfcc_var / 15.0, 0, 1))
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            vocal_signal_2 = float(np.clip(1.0 - np.mean(flatness) * 10, 0, 1))
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            hpr = float(np.mean(y_harmonic ** 2) / (np.mean(y_percussive ** 2) + 1e-8))
            vocal_signal_3 = float(np.clip(hpr / 5.0, 0, 1))
            vocal_presence = (vocal_signal_1 * 0.4 + vocal_signal_2 * 0.3 + vocal_signal_3 * 0.3)
            instrumentalness = float(np.clip(1.0 - vocal_presence, 0.0, 0.95))

            # ── ACOUSTICNESS (inverse of spectral rolloff) ──
            rolloff_mean = float(np.mean(spectral_rolloff))
            acousticness = float(np.clip(1.0 - rolloff_mean / (sr / 2), 0.0, 1.0))

            # ── DANCEABILITY (beat regularity + energy) ──
            beat_regularity = float(np.std(np.diff(beats))) if len(beats) > 2 else 10.0
            danceability = float(np.clip(
                (1 / (1 + beat_regularity / 5)) * 0.6 + rms_mean * 40 * 0.4,
                0.1, 0.99
            ))

            # ── VALENCE (major/minor key + spectral brightness) ──
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
            chroma_mean = np.mean(chroma, axis=1)
            major_score = float(np.dot(chroma_mean, major_profile))
            minor_score = float(np.dot(chroma_mean, minor_profile))
            mode_valence = major_score / (major_score + minor_score + 1e-9)
            brightness = float(np.clip(mean_centroid / 6000.0, 0.0, 1.0))
            valence = float(np.clip((mode_valence * 0.7) + (brightness * 0.3), 0.1, 0.95))

            # ── HARMONIC COMPLEXITY (P3) ──
            # chord_change_rate: fraction of adjacent chroma frames with large L2 delta
            chroma_diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)  # (T-1,)
            chroma_diff_thresh = np.percentile(chroma_diff, 50)            # median threshold
            n_changes = float(np.sum(chroma_diff > chroma_diff_thresh))
            duration_s = float(duration)
            chord_change_rate = float(np.clip(n_changes / (duration_s + 1e-8), 0.0, 20.0))

            # harmonic_complexity: Shannon entropy of mean chroma (normalised to 0-1)
            chroma_normed = chroma_mean / (chroma_mean.sum() + 1e-8)
            chroma_normed = np.clip(chroma_normed, 1e-10, None)
            raw_entropy = float(scipy_entropy(chroma_normed))
            max_entropy = float(np.log(12))  # uniform over 12 pitch classes
            harmonic_complexity = float(np.clip(raw_entropy / (max_entropy + 1e-8), 0.0, 1.0))

            # tonal_stability: how consistently one pitch class dominates each frame
            dominant_pc = np.argmax(chroma, axis=0)  # (T,) — pitch class per frame
            pc_std = float(np.std(dominant_pc))
            # std of uniform RV on {0..11} = 3.45; normalise by that
            tonal_stability = float(np.clip(1.0 - pc_std / 3.46, 0.0, 1.0))

            # ── STRUCTURAL SEGMENTATION (P5) ──
            try:
                from pipelines.structural_segmenter import segment_track
                segments = segment_track(audio_path)
                print(f"[Features] Segments: {[(s['label'], s['start'], s['end']) for s in segments]}")
            except Exception as e:
                print(f"[Features] Segmentation failed: {e}")
                segments = []

            result = {
                "tempo": to_float(tempo),
                "duration": float(duration),
                "key": key,
                "hook_time": round(hook_time, 2) if hook_time is not None else None,
                "hook_confidence": round(hook_conf, 3),
                "hook_time_mert": round(hook_time_mert, 2) if hook_time_mert is not None else None,
                "hook_confidence_mert": round(hook_conf_mert, 3),
                "energy_curve": energy_curve,
                "danceability": round(danceability, 4),
                "energy": round(energy_val, 4),
                "valence": round(valence, 4),
                "acousticness": round(acousticness, 4),
                "instrumentalness": round(instrumentalness, 4),
                "speechiness": round(speechiness, 4),
                "loudness": round(loudness, 4),
                "energy_slope": round(energy_slope, 6),
                "spectral_variance": round(spectral_variance, 4),
                "onset_density": round(onset_density, 4),
                # Harmonic complexity features (P3)
                "chord_change_rate": round(chord_change_rate, 4),
                "harmonic_complexity": round(harmonic_complexity, 4),
                "tonal_stability": round(tonal_stability, 4),
                # Structural segmentation (P5)
                "segments": segments,
            }

            print(f"[Features] {json.dumps({k: v for k, v in result.items() if k != 'energy_curve'}, indent=2)}")
            return result

        except Exception as e:
            print(f"[Features] Extraction failed: {e}")
            traceback.print_exc()
            return {
                "tempo": 120.0, "duration": 60.0, "key": "C",
                "hook_time": None, "hook_confidence": 0.0,
                "hook_time_mert": None, "hook_confidence_mert": 0.0,
                "energy_curve": [0.0] * 6,
                "danceability": 0.5, "energy": 0.5, "valence": 0.5,
                "acousticness": 0.5, "instrumentalness": 0.1,
                "speechiness": 0.05, "loudness": -10.0,
                "energy_slope": 0.0, "spectral_variance": 0.2,
                "onset_density": 0.3,
                # Harmonic complexity fallbacks
                "chord_change_rate": 2.0,
                "harmonic_complexity": 0.5,
                "tonal_stability": 0.5,
                # Structural segmentation fallback
                "segments": [],
            }
