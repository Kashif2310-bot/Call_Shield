$backend = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$frontend = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1

if ($backend) {
    try {
        Stop-Process -Id $backend.OwningProcess -Force -ErrorAction Stop
        Write-Host "Stopped backend process on 8001 (PID $($backend.OwningProcess))."
    } catch {
        Write-Host "Backend process on 8001 was not stoppable."
    }
} else {
    Write-Host "No backend process listening on 8001."
}

if ($frontend) {
    try {
        Stop-Process -Id $frontend.OwningProcess -Force -ErrorAction Stop
        Write-Host "Stopped frontend process on 8501 (PID $($frontend.OwningProcess))."
    } catch {
        Write-Host "Frontend process on 8501 was not stoppable."
    }
} else {
    Write-Host "No frontend process listening on 8501."
}
