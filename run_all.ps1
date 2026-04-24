param(
    [int]$BackendPort = 8001,
    [int]$FrontendPort = 8501
)

$apiBase = "http://127.0.0.1:$BackendPort"
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\run_backend.ps1 -Port $BackendPort" -PassThru
Start-Sleep -Seconds 2
$frontend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\run_frontend.ps1 -ApiBase '$apiBase' -Port $FrontendPort" -PassThru

Write-Host "Started CallShield services."
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Backend: $apiBase"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
