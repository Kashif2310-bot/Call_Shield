from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from ..schemas import ContextRiskResponse

load_dotenv()


def _heuristic_context_analysis(transcript: str) -> ContextRiskResponse:
    text = (transcript or "").lower()

    urgency_terms = ["urgent", "immediately", "now", "asap", "right away", "emergency"]
    financial_terms = ["bank", "transfer", "upi", "otp", "payment", "money", "account", "wallet"]
    emotional_terms = ["please", "help me", "family", "hospital", "panic", "don't tell anyone"]

    urgency = min(1.0, sum(term in text for term in urgency_terms) / 3.0)
    financial_intent = min(1.0, sum(term in text for term in financial_terms) / 3.0)
    emotional = min(1.0, sum(term in text for term in emotional_terms) / 3.0)

    explanation = "Heuristic analysis used (no LLM configured)."
    return ContextRiskResponse(
        urgency=float(urgency),
        financial_intent=float(financial_intent),
        emotional_manipulation=float(emotional),
        explanation=explanation,
    )


def _parse_llm_json(payload: str) -> Dict[str, Any]:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(payload[start : end + 1])


def analyze_context(transcript: str) -> ContextRiskResponse:
    transcript = transcript or ""
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")

    if not api_key or not transcript.strip():
        return _heuristic_context_analysis(transcript)

    prompt = (
        "You are a scam call risk classifier. Return ONLY JSON with keys: "
        "urgency, financial_intent, emotional_manipulation, explanation. "
        "Scores must be floats between 0 and 1.\n\nTranscript:\n"
        f"{transcript}"
    )

    try:
        resp = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Respond with strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content)
        return ContextRiskResponse(
            urgency=float(max(0.0, min(1.0, parsed.get("urgency", 0.0)))),
            financial_intent=float(max(0.0, min(1.0, parsed.get("financial_intent", 0.0)))),
            emotional_manipulation=float(
                max(0.0, min(1.0, parsed.get("emotional_manipulation", 0.0)))
            ),
            explanation=str(parsed.get("explanation", "LLM analysis completed.")),
        )
    except Exception:
        return _heuristic_context_analysis(transcript)
