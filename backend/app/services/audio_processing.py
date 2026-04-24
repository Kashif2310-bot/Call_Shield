from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Tuple
import sys
import types

import numpy as np
import soundfile as sf

try:
    import numba as _numba  # noqa: F401
except Exception:
    # Corporate endpoint policies can block numba native extensions.
    # Provide a tiny shim so librosa can still run for prototype usage.
    shim = types.ModuleType("numba")

    def _passthrough_decorator(*args, **kwargs):
        def _wrapper(func):
            return func

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return _wrapper

    shim.jit = _passthrough_decorator
    shim.njit = _passthrough_decorator
    shim.vectorize = _passthrough_decorator
    shim.guvectorize = _passthrough_decorator
    shim.stencil = _passthrough_decorator
    shim.prange = range
    sys.modules["numba"] = shim

import librosa


@dataclass
class AudioFeatures:
    features: Dict[str, float]
    mfcc_mean: np.ndarray
    mfcc_matrix: np.ndarray
    embedding: np.ndarray
    duration_seconds: float
    sample_rate: int


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(BytesIO(audio_bytes), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y=y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def extract_audio_features(y: np.ndarray, sr: int) -> AudioFeatures:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # Manual zero-crossing rate to avoid numba-dependent librosa path.
    zc = np.sum(np.abs(np.diff(np.signbit(y).astype(np.int8))))
    zcr_mean = float(zc) / max(1.0, float(len(y)))
    rms = librosa.feature.rms(y=y)

    # Lightweight pitch estimate via FFT peak tracking.
    frame_len = int(0.04 * sr)
    hop_len = int(0.02 * sr)
    if frame_len <= 0 or hop_len <= 0 or len(y) < frame_len:
        pitch_values = np.array([], dtype=np.float32)
    else:
        pitches_list = []
        window = np.hanning(frame_len)
        for start in range(0, len(y) - frame_len + 1, hop_len):
            frame = y[start : start + frame_len] * window
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
            valid = (freqs >= 50.0) & (freqs <= 400.0)
            if np.any(valid):
                peak_idx = np.argmax(spectrum[valid])
                peak_freq = freqs[valid][peak_idx]
                if spectrum[valid][peak_idx] > 1e-6:
                    pitches_list.append(float(peak_freq))
        pitch_values = np.array(pitches_list, dtype=np.float32)

    mean_pitch = float(np.mean(pitch_values)) if pitch_values.size else 0.0
    pitch_std = float(np.std(pitch_values)) if pitch_values.size else 0.0

    mfcc_mean = np.mean(mfcc, axis=1)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Lightweight speaker embedding surrogate for prototype:
    # Concatenate MFCC and deltas to form a fixed-length vector.
    embedding = np.concatenate(
        [
            mfcc_mean,
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1),
            np.array(
                [
                    float(np.mean(spectral_centroid)),
                    float(np.mean(spectral_bandwidth)),
                    float(np.mean(spectral_rolloff)),
                    zcr_mean,
                    float(np.mean(rms)),
                    mean_pitch,
                    pitch_std,
                ]
            ),
        ]
    ).astype(np.float32)

    features = {
        "mfcc_variance": float(np.mean(np.var(mfcc, axis=1))),
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "zcr_mean": zcr_mean,
        "rms_mean": float(np.mean(rms)),
    }

    return AudioFeatures(
        features=features,
        mfcc_mean=mfcc_mean.astype(np.float32),
        mfcc_matrix=mfcc.astype(np.float32),
        embedding=embedding,
        duration_seconds=float(len(y) / sr),
        sample_rate=sr,
    )
