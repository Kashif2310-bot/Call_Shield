from __future__ import annotations

import io
import math
import os
import wave

import requests

API = os.getenv("CALLSHIELD_API_BASE", "http://127.0.0.1:8000")


def _sine_wav_bytes(freq: float = 220.0, duration: float = 2.0, sr: int = 16000) -> bytes:
    n = int(duration * sr)
    frames = bytearray()
    for i in range(n):
        val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * (i / sr)))
        frames += int(val).to_bytes(2, byteorder="little", signed=True)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(frames))
    return bio.getvalue()


def main() -> None:
    wav_a = _sine_wav_bytes(freq=220.0)
    wav_b = _sine_wav_bytes(freq=235.0)

    analyze = requests.post(
        f"{API}/analyze_audio",
        files={"audio_file": ("sample.wav", wav_a, "audio/wav")},
        timeout=20,
    )
    analyze.raise_for_status()
    a = analyze.json()
    print("analyze_audio:", {k: a[k] for k in ["anomaly_score", "authenticity_score", "confidence"]})

    compare = requests.post(
        f"{API}/compare_voice",
        files={
            "reference_audio": ("ref.wav", wav_a, "audio/wav"),
            "test_audio": ("test.wav", wav_b, "audio/wav"),
        },
        timeout=20,
    )
    compare.raise_for_status()
    c = compare.json()
    print("compare_voice:", c)

    risk = requests.post(
        f"{API}/risk_score",
        json={
            "anomaly_score": a["anomaly_score"],
            "identity_match_score": c["identity_match_score"],
            "transcript": "This is urgent. Transfer money now. Do not tell anyone.",
        },
        timeout=20,
    )
    risk.raise_for_status()
    print("risk_score:", risk.json())

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
