from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from .schemas import (
    AudioAnalysisResponse,
    RiskScoreRequest,
    RiskScoreResponse,
    VoiceComparisonResponse,
)
from .services.audio_processing import extract_audio_features, load_audio_from_bytes
from .services.context_analysis import analyze_context
from .services.scoring import (
    aggregate_trust_score,
    compare_embeddings,
    compute_anomaly_score,
    compute_authenticity_score,
    compute_confidence,
)

app = FastAPI(title="CallShield API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze_audio", response_model=AudioAnalysisResponse)
async def analyze_audio(audio_file: UploadFile = File(...)) -> AudioAnalysisResponse:
    try:
        audio_bytes = await audio_file.read()
        y, sr = load_audio_from_bytes(audio_bytes)
        extracted = extract_audio_features(y, sr)
        anomaly = compute_anomaly_score(extracted.features)
        authenticity = compute_authenticity_score(anomaly)
        confidence = compute_confidence(extracted.features)

        return AudioAnalysisResponse(
            duration_seconds=extracted.duration_seconds,
            sample_rate=extracted.sample_rate,
            anomaly_score=anomaly,
            authenticity_score=authenticity,
            confidence=confidence,
            features=extracted.features,
            mfcc_mean=extracted.mfcc_mean.tolist(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio analysis failed: {exc}") from exc


@app.post("/compare_voice", response_model=VoiceComparisonResponse)
async def compare_voice(
    reference_audio: UploadFile = File(...), test_audio: UploadFile = File(...)
) -> VoiceComparisonResponse:
    try:
        ref_bytes = await reference_audio.read()
        tst_bytes = await test_audio.read()
        y_ref, sr_ref = load_audio_from_bytes(ref_bytes)
        y_tst, sr_tst = load_audio_from_bytes(tst_bytes)
        ref_feat = extract_audio_features(y_ref, sr_ref)
        tst_feat = extract_audio_features(y_tst, sr_tst)
        result = compare_embeddings(ref_feat.embedding, tst_feat.embedding)
        return VoiceComparisonResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Voice comparison failed: {exc}") from exc


@app.post("/risk_score", response_model=RiskScoreResponse)
def risk_score(payload: RiskScoreRequest) -> RiskScoreResponse:
    context = analyze_context(payload.transcript or "")
    scored = aggregate_trust_score(payload.anomaly_score, payload.identity_match_score, context)
    return RiskScoreResponse(
        trust_score=float(scored["trust_score"]),
        alert=str(scored["alert"]),
        context=context,
        components={
            "anomaly_score": payload.anomaly_score,
            "identity_match_score": payload.identity_match_score,
            "context_risk": float(scored["context_risk"]),
        },
    )
