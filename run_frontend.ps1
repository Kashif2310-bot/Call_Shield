param(
    [string]$ApiBase = "http://127.0.0.1:8001",
    [int]$Port = 8501
)

$env:CALLSHIELD_API_BASE = $ApiBase
python -m streamlit run frontend/streamlit_app.py --server.port $Port
