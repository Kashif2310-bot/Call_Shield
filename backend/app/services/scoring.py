from __future__ import annotations

from typing import Dict

import numpy as np

from ..schemas import ContextRiskResponse


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def compute_anomaly_score(features: Dict[str, float]) -> float:
    # Heuristic indicators often seen in synthetic/cloned voices.
    pitch_std = features.get("pitch_std", 0.0)
    zcr_mean = features.get("zcr_mean", 0.0)
    mfcc_variance = features.get("mfcc_variance", 0.0)
    spectral_bandwidth = features.get("spectral_bandwidth_mean", 0.0)
    rms_mean = features.get("rms_mean", 0.0)

    pitch_flatness = max(0.0, 50.0 - pitch_std) / 50.0
    zcr_irregularity = max(0.0, 0.08 - zcr_mean) / 0.08
    mfcc_smoothness = max(0.0, 120.0 - mfcc_variance) / 120.0
    bandwidth_narrowness = max(0.0, 1800.0 - spectral_bandwidth) / 1800.0
    energy_artifact = max(0.0, 0.015 - rms_mean) / 0.015

    weighted = (
        0.28 * pitch_flatness
        + 0.18 * zcr_irregularity
        + 0.24 * mfcc_smoothness
        + 0.18 * bandwidth_narrowness
        + 0.12 * energy_artifact
    )
    return float(np.clip(weighted, 0.0, 1.0))


def compute_authenticity_score(anomaly_score: float) -> float:
    return float(np.clip(1.0 - anomaly_score, 0.0, 1.0))


def compute_confidence(features: Dict[str, float]) -> float:
    quality = 1.0
    if features.get("rms_mean", 0.0) < 0.005:
        quality -= 0.25
    if features.get("spectral_centroid_mean", 0.0) < 300:
        quality -= 0.15
    if features.get("mean_pitch", 0.0) <= 0:
        quality -= 0.15
    return float(np.clip(quality, 0.4, 1.0))


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float | bool]:
    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    similarity = float(np.dot(emb1, emb2) / denom) if denom > 0 else 0.0
    match_score = float(np.clip((similarity + 1) / 2, 0.0, 1.0))
    is_match = match_score >= 0.72
    return {
        "cosine_similarity": similarity,
        "identity_match_score": match_score,
        "is_match": is_match,
    }


def aggregate_trust_score(
    anomaly_score: float, identity_match_score: float, context: ContextRiskResponse
) -> Dict[str, float | str]:
    context_risk = (
        0.4 * context.urgency
        + 0.35 * context.financial_intent
        + 0.25 * context.emotional_manipulation
    )
    trust_raw = (
        0.45 * (1.0 - anomaly_score)
        + 0.35 * identity_match_score
        + 0.20 * (1.0 - context_risk)
    )
    # Smooth extremes while retaining 0-100 scale readability.
    trust_score = float(np.clip(100.0 * _sigmoid(4 * (trust_raw - 0.5)), 0.0, 100.0))

    if trust_score >= 70:
        alert = "Safe"
    elif trust_score >= 40:
        alert = "Suspicious"
    else:
        alert = "Scam"

    return {
        "trust_score": trust_score,
        "alert": alert,
        "context_risk": float(context_risk),
    }
