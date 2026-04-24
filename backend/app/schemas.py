from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AudioAnalysisResponse(BaseModel):
    duration_seconds: float
    sample_rate: int
    anomaly_score: float = Field(ge=0, le=1)
    authenticity_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    features: Dict[str, float]
    mfcc_mean: List[float]


class VoiceComparisonResponse(BaseModel):
    cosine_similarity: float = Field(ge=-1, le=1)
    identity_match_score: float = Field(ge=0, le=1)
    is_match: bool


class RiskScoreRequest(BaseModel):
    anomaly_score: float = Field(ge=0, le=1)
    identity_match_score: float = Field(ge=0, le=1)
    transcript: Optional[str] = ""


class ContextRiskResponse(BaseModel):
    urgency: float = Field(ge=0, le=1)
    financial_intent: float = Field(ge=0, le=1)
    emotional_manipulation: float = Field(ge=0, le=1)
    explanation: str


class RiskScoreResponse(BaseModel):
    trust_score: float = Field(ge=0, le=100)
    alert: str
    context: ContextRiskResponse
    components: Dict[str, float]
