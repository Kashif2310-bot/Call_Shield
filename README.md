# CallShield Prototype

CallShield is a local, lightweight prototype for real-time audio deepfake detection and voice trust analysis.

## Features

- FastAPI backend for audio analysis and scoring.
- Librosa-based feature extraction:
  - MFCC
  - pitch statistics
  - spectral features
- Synthetic audio anomaly scoring via heuristics (no heavy training).
- Speaker identity matching using cosine similarity over lightweight embeddings.
- Context risk analysis using optional LLM API, with heuristic fallback.
- Streamlit frontend with:
  - audio upload
  - microphone recording (via `streamlit-audiorecorder`)
  - waveform and MFCC heatmap
  - authenticity, identity match, trust score, and alert state
- Real-time simulation by processing 1-2 second chunks.

## Project Structure

```text
micx/
├── backend/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── services/
│           ├── audio_processing.py
│           ├── context_analysis.py
│           └── scoring.py
├── frontend/
│   └── streamlit_app.py
├── .env.example
├── requirements.txt
└── README.md
```

## API Endpoints

- `POST /analyze_audio`  
  Input: one audio file  
  Output: extracted features, anomaly/authenticity/confidence scores

- `POST /compare_voice`  
  Input: `reference_audio`, `test_audio`  
  Output: cosine similarity and identity match score

- `POST /risk_score`  
  Input: `anomaly_score`, `identity_match_score`, `transcript`  
  Output: trust score (0-100), alert (`Safe`, `Suspicious`, `Scam`) and context signals

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure optional LLM support:

```bash
copy .env.example .env
```

Set `LLM_API_KEY` in `.env` if you want LLM-based transcript analysis.  
Without API credentials, heuristic context analysis is used automatically.

## Run Backend

```bash
uvicorn backend.app.main:app --reload
```

or on Windows PowerShell:

```powershell
.\run_backend.ps1
```

Backend URL (default): `http://127.0.0.1:8001`

## Run Frontend

In a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

or:

```powershell
.\run_frontend.ps1
```

## Run Everything

On Windows PowerShell, launch backend + frontend together:

```powershell
.\run_all.ps1
```

## Recommended Safe Launch (Windows)

For environments with strict DLL policies, use:

```powershell
.\run_safe.ps1
```

This starts:
- backend on `http://127.0.0.1:8001`
- frontend on `http://127.0.0.1:8501`

To stop both default ports:

```powershell
.\stop_all.ps1
```

## Quick Smoke Test

After backend is running:

```bash
python scripts/smoke_test.py
```

or against a specific backend:

```powershell
$env:CALLSHIELD_API_BASE="http://127.0.0.1:8001"; python scripts/smoke_test.py
```

This verifies `/analyze_audio`, `/compare_voice`, and `/risk_score` with generated test audio.

## Notes

- This is a prototype focused on concept demonstration.
- The speaker embedding is lightweight and heuristic-driven.
- For production, replace heuristic scoring with calibrated models and add robust ASR + anti-spoof model pipelines.
