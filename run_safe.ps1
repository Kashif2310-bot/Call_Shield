param(
    [int]$BackendPort = 8001,
    [int]$FrontendPort = 8501
)

Write-Host "Launching CallShield in safe local mode..."
Write-Host "Using backend port $BackendPort and frontend port $FrontendPort"
.\run_all.ps1 -BackendPort $BackendPort -FrontendPort $FrontendPort
