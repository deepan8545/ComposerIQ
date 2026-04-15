"""
Structural Segmenter (P5)
--------------------------
Detects verse/chorus/bridge/intro/outro sections using librosa's
recurrence matrix and agglomerative clustering on Mel spectral features.

Pure librosa — no new dependencies.

Usage:
    from pipelines.structural_segmenter import segment_track
    segments = segment_track("/path/to/audio.mp3")
    # → [{"label": "intro", "start": 0.0, "end": 14.3}, ...]

Algorithm:
  1. Compute Mel spectrogram (128 bands, hop=512)
  2. Beat-synchronise the spectrogram (frame → beat-level)
  3. Build a recurrence matrix from beat-sync MFCCs + chroma
  4. Compute the Laplacian eigenvectors → structural boundary positions
  5. Merge segments shorter than min_seg_sec into neighbours
  6. Heuristically label segments by their energy profile and position
"""
from __future__ import annotations

import numpy as np
import librosa
from typing import List, Dict, Optional


def segment_track(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    n_segments: int = 8,
    min_seg_sec: float = 8.0,
) -> List[Dict]:
    """
    Detect structural segments in an audio track.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    sr : int
        Sample rate for analysis.
    hop_length : int
        Librosa hop length.
    n_segments : int
        Maximum number of segments to attempt.
    min_seg_sec : float
        Merge segments shorter than this into adjacent ones.

    Returns
    -------
    List[Dict]
        A list of segment dicts, each with keys:
        {"label": str, "start": float, "end": float, "energy_mean": float}
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # ---- Feature extraction ----
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
        chroma    = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        rms       = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Beat-synchronise features → more stable segments
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        if len(beat_frames) < 4:
            # Not enough beats — fall back to time-based segmentation
            return _fallback_segments(duration)

        mfcc_sync   = librosa.util.sync(mfcc,   beat_frames, aggregate=np.median)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        rms_sync    = librosa.util.sync(rms.reshape(1, -1), beat_frames, aggregate=np.mean)[0]

        # Stack features
        features_sync = np.vstack([
            librosa.util.normalize(mfcc_sync,   norm=2),
            librosa.util.normalize(chroma_sync, norm=2),
        ])

        # ---- Recurrence matrix + Laplacian boundaries ----
        R = librosa.segment.recurrence_matrix(
            features_sync,
            width=3,
            mode="affinity",
            sym=True,
        )

        # Combine recurrence with linear decay for structure detection
        df = librosa.segment.timelag_filter(librosa.decompose.nn_filter)(R)
        R_hat = np.maximum(R, df)

        # Laplacian eigenvectors
        n_segs_safe = min(n_segments, features_sync.shape[1] - 1, 6)
        _, evecs = librosa.segment.agglomerative(R_hat, n_segs_safe)

        # Boundary frames (in beat-sync space)
        bounds_beat = np.concatenate([[0], evecs, [features_sync.shape[1]]])
        bounds_beat = np.sort(np.unique(bounds_beat.astype(int)))

        # Convert beat frames → time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        n_beats = len(beat_times)

        def beat_to_time(b: int) -> float:
            if b <= 0:
                return 0.0
            if b >= n_beats:
                return float(duration)
            return float(beat_times[min(b, n_beats - 1)])

        raw_segments = []
        for i in range(len(bounds_beat) - 1):
            start = beat_to_time(bounds_beat[i])
            end   = beat_to_time(bounds_beat[i + 1])
            if end - start < 0.5:
                continue

            # Mean RMS energy for this segment (from beat-synced RMS)
            b_start = int(bounds_beat[i])
            b_end   = int(bounds_beat[i + 1])
            seg_rms = float(np.mean(rms_sync[b_start:b_end])) if b_end > b_start else 0.0

            raw_segments.append({
                "start":       round(start, 2),
                "end":         round(end, 2),
                "energy_mean": round(seg_rms, 6),
            })

        if not raw_segments:
            return _fallback_segments(duration)

        # ---- Merge short segments ----
        merged = _merge_short(raw_segments, min_sec=min_seg_sec)

        # ---- Label segments heuristically ----
        labeled = _label_segments(merged, duration)

        return labeled

    except Exception as e:
        print(f"[Segmenter] Failed: {e}")
        return _fallback_segments(duration if 'duration' in dir() else 180.0)


def _merge_short(segments: List[Dict], min_sec: float) -> List[Dict]:
    """Merge segments shorter than min_sec into their longer neighbour."""
    if not segments:
        return segments

    result = list(segments)
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(result):
            seg = result[i]
            if (seg["end"] - seg["start"]) < min_sec and len(result) > 1:
                # Merge into the next segment (or previous if last)
                if i + 1 < len(result):
                    result[i + 1]["start"] = seg["start"]
                    result[i + 1]["energy_mean"] = (
                        seg["energy_mean"] + result[i + 1]["energy_mean"]
                    ) / 2.0
                elif new:
                    new[-1]["end"] = seg["end"]
                    new[-1]["energy_mean"] = (
                        new[-1]["energy_mean"] + seg["energy_mean"]
                    ) / 2.0
                    i += 1
                    continue
                changed = True
                i += 1
                continue
            new.append(seg)
            i += 1
        result = new

    return result


def _label_segments(segments: List[Dict], duration: float) -> List[Dict]:
    """
    Heuristically label segments using position + energy profile.

    Rules:
      - First segment (start < 10s): intro
      - Last segment (end > duration-20s): outro
      - Highest-energy segments: chorus
      - Mid-energy + early-mid position: verse
      - Short high-energy transition segments: bridge
      - Remaining: verse
    """
    if not segments:
        return segments

    n = len(segments)
    energies = np.array([s["energy_mean"] for s in segments])
    if energies.max() > 0:
        energies_n = energies / (energies.max() + 1e-8)
    else:
        energies_n = np.zeros(n)

    energy_thresh_chorus = 0.75
    energy_thresh_verse  = 0.35

    labeled = []
    for i, seg in enumerate(segments):
        e_norm = float(energies_n[i])
        s_time = seg["start"]
        e_time = seg["end"]
        seg_len = e_time - s_time

        if i == 0 and s_time < 10.0:
            label = "intro"
        elif i == n - 1 and e_time > duration - 20.0:
            label = "outro"
        elif e_norm >= energy_thresh_chorus:
            label = "chorus"
        elif seg_len < 15.0 and e_norm >= energy_thresh_verse and i > 0:
            label = "bridge"
        else:
            label = "verse"

        labeled.append({
            "label":       label,
            "start":       seg["start"],
            "end":         seg["end"],
            "energy_mean": seg["energy_mean"],
        })

    return labeled


def _fallback_segments(duration: float) -> List[Dict]:
    """Return simple evenly-spaced fallback segments when detection fails."""
    third = duration / 3.0
    return [
        {"label": "intro",  "start": 0.0,         "end": round(third, 1),         "energy_mean": 0.0},
        {"label": "verse",  "start": round(third, 1), "end": round(2 * third, 1), "energy_mean": 0.0},
        {"label": "chorus", "start": round(2 * third, 1), "end": round(duration, 1), "energy_mean": 0.0},
    ]
