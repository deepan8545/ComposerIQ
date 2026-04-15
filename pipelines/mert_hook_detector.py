"""
Frame-Level MERT Hook Detector (P2)
-------------------------------------
Uses MERT's frame-level hidden states to locate the hook more accurately
than librosa onset thresholds.

Strategy:
  1. Extract per-frame 768-dim MERT embeddings for the first 60% of the track
  2. Compute the L2-norm of the frame-to-frame ΔH (hidden state delta)
     — large delta = sudden change in sonic character (key hook indicator)
  3. Smooth the delta curve with a Gaussian window to avoid spurious spikes
  4. Find the first frame in the smoothed curve that exceeds the 80th percentile
     AND is at least 2s into the track
  5. Fuse with the librosa estimate: if they agree within 5s → high confidence;
     otherwise use the MERT estimate (more musically grounded)

Returns (hook_time_seconds, confidence) where confidence is 0-1.
"""
from __future__ import annotations

import numpy as np
import librosa
from typing import Optional, Tuple


def detect_hook_mert(
    audio_path: str,
    embedder,                  # MERTEmbedder singleton from mert_embedder.py
    librosa_hook_time: Optional[float] = None,
    analysis_window_sec: float = 90.0,
    min_hook_sec: float = 2.0,
    delta_pct_threshold: float = 80.0,
    agreement_window_sec: float = 5.0,
) -> Tuple[Optional[float], float]:
    """
    Detect hook using MERT frame-level hidden state deltas.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    embedder : MERTEmbedder
        Singleton MERT model (already loaded, won't reload).
    librosa_hook_time : float, optional
        Existing librosa-based estimate for fusion.
    analysis_window_sec : float
        Only analyse up to this many seconds (avoid analysing full track).
    min_hook_sec : float
        Ignore any hook candidates before this time (skip intros).
    delta_pct_threshold : float
        Percentile threshold on the delta curve for candidate detection.
    agreement_window_sec : float
        Agreement window (seconds) for fusing MERT + librosa estimates.

    Returns
    -------
    (hook_time, confidence) : Tuple[Optional[float], float]
        hook_time   — detected hook in seconds, or None if detection failed
        confidence  — 0.0-1.0 (0.9+ = high agreement, 0.6 = MERT only, 0.3 = fallback)
    """
    try:
        import torch

        embedder._load()
        model     = embedder._model
        processor = embedder._processor

        # Load at MERT's native sr (24 kHz)
        y, sr = librosa.load(audio_path, sr=24000, mono=True)
        total_sec = len(y) / sr

        # Analyse first 60% of track, capped at analysis_window_sec
        window_sec = min(analysis_window_sec, total_sec * 0.60)
        y_window   = y[: int(window_sec * sr)]

        # ---- Frame MERT in 1-second hops ----
        hop_sec    = 1.0
        frame_len  = int(1.5 * sr)   # 1.5s frames (MERT needs some context)
        hop_frames = int(hop_sec * sr)

        frame_embeddings = []
        frame_times      = []

        pos = 0
        while pos + frame_len <= len(y_window):
            segment = y_window[pos: pos + frame_len]
            inputs  = processor(segment, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            # Mean-pool the last hidden layer over time → 768-dim vector
            hidden = out.hidden_states[-1]              # (1, T, 768)
            vec    = hidden.mean(dim=1).squeeze().numpy()  # (768,)
            frame_embeddings.append(vec.astype(np.float32))
            t = pos / sr + frame_len / (2 * sr)        # centre time of frame
            frame_times.append(t)
            pos += hop_frames

        if len(frame_embeddings) < 3:
            return librosa_hook_time, 0.3              # not enough frames

        E = np.stack(frame_embeddings)                 # (N, 768)

        # ---- Frame-to-frame L2 delta ----
        deltas     = np.linalg.norm(np.diff(E, axis=0), axis=1)  # (N-1,)
        delta_times = np.array(frame_times[1:])

        # Smooth with a 3-frame Gaussian window
        kernel  = np.array([0.25, 0.5, 0.25])
        if len(deltas) >= 3:
            smoothed = np.convolve(deltas, kernel, mode="same")
        else:
            smoothed = deltas

        # ---- Find first dominant peak beyond min_hook_sec ----
        threshold = np.percentile(smoothed, delta_pct_threshold)
        valid     = (delta_times >= min_hook_sec) & (smoothed > threshold)
        candidates = np.where(valid)[0]

        if len(candidates) == 0:
            # Relax threshold if nothing found
            threshold  = np.percentile(smoothed, 60)
            valid      = (delta_times >= min_hook_sec) & (smoothed > threshold)
            candidates = np.where(valid)[0]

        if len(candidates) == 0:
            return librosa_hook_time, 0.3

        mert_hook_time = float(delta_times[candidates[0]])

        # ---- Normalised confidence from delta magnitude ----
        peak_delta = float(smoothed[candidates[0]])
        norm_conf  = float(np.clip(peak_delta / (np.max(smoothed) + 1e-8), 0, 1))
        mert_conf  = float(np.clip(0.5 + norm_conf * 0.4, 0.5, 0.9))  # 0.5-0.9 range

        # ---- Fuse with librosa estimate ----
        if librosa_hook_time is not None:
            diff = abs(mert_hook_time - librosa_hook_time)
            if diff <= agreement_window_sec:
                # Agreement — blend times, high confidence
                fused_time = (mert_hook_time * 0.6 + librosa_hook_time * 0.4)
                confidence = min(0.95, mert_conf + 0.15)  # boost for agreement
                return round(fused_time, 2), round(confidence, 3)
            else:
                # Disagreement — trust MERT but lower confidence slightly
                return round(mert_hook_time, 2), round(mert_conf * 0.9, 3)

        return round(mert_hook_time, 2), round(mert_conf, 3)

    except Exception as e:
        print(f"[MERTHook] Detection failed: {e}")
        return librosa_hook_time, 0.3
