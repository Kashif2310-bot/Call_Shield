from __future__ import annotations

import io
import os
import time
from typing import Optional
import sys
import types

import numpy as np
import requests
import soundfile as sf
import streamlit as st

try:
    import numba as _numba  # noqa: F401
except Exception:
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

try:
    from audiorecorder import audiorecorder
except Exception:
    audiorecorder = None

DEFAULT_API_BASE = os.getenv("CALLSHIELD_API_BASE", "http://127.0.0.1:8000")


def _to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, y, sr, format="WAV")
    return bio.getvalue()


def _extract_waveform_and_mfcc(audio_bytes: bytes) -> tuple[np.ndarray, int, np.ndarray]:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return y, sr, mfcc


def _plot_waveform(y: np.ndarray, sr: int) -> None:
    duration = len(y) / sr
    t = np.linspace(0.0, duration, num=len(y), endpoint=False)
    stride = max(1, len(y) // 4000)
    st.line_chart(
        {
            "time_s": t[::stride],
            "amplitude": y[::stride],
        },
        x="time_s",
        y="amplitude",
        use_container_width=True,
    )


def _plot_mfcc(mfcc: np.ndarray) -> None:
    # Render a lightweight heatmap image without matplotlib.
    m = mfcc.astype(np.float32)
    m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-8)
    r = (255 * m).astype(np.uint8)
    g = (255 * np.sqrt(m)).astype(np.uint8)
    b = (255 * (1.0 - m)).astype(np.uint8)
    heatmap = np.stack([r, g, b], axis=2)
    st.image(heatmap, caption="MFCC Heatmap", use_container_width=True)


def _analyze_chunk(api_base: str, chunk_bytes: bytes) -> Optional[dict]:
    try:
        files = {"audio_file": ("chunk.wav", chunk_bytes, "audio/wav")}
        r = requests.post(f"{api_base}/analyze_audio", files=files, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _compare_voice(api_base: str, reference_bytes: bytes, test_bytes: bytes) -> dict:
    files = {
        "reference_audio": ("ref.wav", reference_bytes, "audio/wav"),
        "test_audio": ("test.wav", test_bytes, "audio/wav"),
    }
    r = requests.post(f"{api_base}/compare_voice", files=files, timeout=30)
    r.raise_for_status()
    return r.json()


def _risk_score(
    api_base: str, anomaly_score: float, identity_match_score: float, transcript: str
) -> dict:
    payload = {
        "anomaly_score": anomaly_score,
        "identity_match_score": identity_match_score,
        "transcript": transcript,
    }
    r = requests.post(f"{api_base}/risk_score", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(page_title="CallShield", layout="wide")
    st.title("CallShield: Real-Time Audio Deepfake & Voice Trust Analyzer")

    with st.sidebar:
        st.subheader("Configuration")
        api_base = st.text_input("Backend API URL", DEFAULT_API_BASE).strip() or DEFAULT_API_BASE
        chunk_sec = st.slider("Chunk Size (seconds)", 1, 2, 1)
        st.caption("Runs live simulation by processing chunks sequentially.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Incoming Call Audio")
        uploaded_audio = st.file_uploader(
            "Upload suspicious call audio", type=["wav", "mp3", "ogg", "m4a"]
        )

        recorded_bytes = None
        if audiorecorder:
            st.write("Or record from microphone:")
            rec = audiorecorder("Start Recording", "Stop Recording")
            if len(rec) > 0:
                recorded_bytes = rec.export(format="wav").read()
                st.audio(recorded_bytes, format="audio/wav")
        else:
            st.info("Install streamlit-audiorecorder for in-app recording.")

    with col2:
        st.subheader("Reference Voice (Optional)")
        reference_audio = st.file_uploader(
            "Upload trusted reference voice",
            type=["wav", "mp3", "ogg", "m4a"],
            key="reference",
        )

    transcript = st.text_area(
        "Call Transcript (for context analysis)",
        placeholder="Paste transcript here to assess urgency, financial intent, and manipulation...",
        height=120,
    )

    if st.button("Run CallShield Analysis", type="primary"):
        main_audio = uploaded_audio.read() if uploaded_audio else recorded_bytes
        if not main_audio:
            st.error("Upload or record call audio first.")
            return

        y, sr, mfcc = _extract_waveform_and_mfcc(main_audio)
        _plot_waveform(y, sr)
        _plot_mfcc(mfcc)

        status = st.empty()
        prog = st.progress(0)
        metrics_box = st.container()

        samples_per_chunk = int(chunk_sec * sr)
        n_chunks = max(1, int(np.ceil(len(y) / samples_per_chunk)))
        anomalies = []
        authenticities = []

        status.info("Starting chunk-level real-time simulation...")
        for idx in range(n_chunks):
            start = idx * samples_per_chunk
            end = min(len(y), (idx + 1) * samples_per_chunk)
            chunk = y[start:end]
            chunk_bytes = _to_wav_bytes(chunk, sr)
            res = _analyze_chunk(api_base, chunk_bytes)
            if res:
                anomalies.append(res["anomaly_score"])
                authenticities.append(res["authenticity_score"])

            prog.progress((idx + 1) / n_chunks)
            status.info(f"Processed chunk {idx + 1}/{n_chunks}")
            time.sleep(0.12)

        avg_anomaly = float(np.mean(anomalies)) if anomalies else 0.5
        avg_authenticity = float(np.mean(authenticities)) if authenticities else 0.5

        identity_match_score = 0.5
        if reference_audio:
            try:
                ref_bytes = reference_audio.read()
                cmp_res = _compare_voice(api_base, ref_bytes, main_audio)
                identity_match_score = cmp_res["identity_match_score"]
            except Exception as exc:
                st.warning(f"Voice comparison failed, using neutral identity score: {exc}")

        risk = _risk_score(api_base, avg_anomaly, identity_match_score, transcript)

        with metrics_box:
            a, b, c = st.columns(3)
            a.metric("Voice Authenticity", f"{avg_authenticity * 100:.1f}%")
            b.metric("Identity Match", f"{identity_match_score * 100:.1f}%")
            c.metric("Final Trust Score", f"{risk['trust_score']:.1f}/100")

            alert = risk["alert"]
            if alert == "Safe":
                st.success("Alert: Safe")
            elif alert == "Suspicious":
                st.warning("Alert: Suspicious")
            else:
                st.error("Alert: Scam")

            st.write("### Context Signals")
            st.write(
                f"- Urgency: {risk['context']['urgency'] * 100:.1f}%\n"
                f"- Financial intent: {risk['context']['financial_intent'] * 100:.1f}%\n"
                f"- Emotional manipulation: {risk['context']['emotional_manipulation'] * 100:.1f}%"
            )
            st.caption(risk["context"]["explanation"])

        status.success("Analysis complete.")


if __name__ == "__main__":
    main()
