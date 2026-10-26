# scripts/start_viewer.ps1
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location "$repoRoot/viewer"

$logFile = "$repoRoot/logs/viewer_dev.log"
$pidFile = "$repoRoot/logs/viewer_dev.pid"

if (-not (Test-Path "$repoRoot/logs")) {
    New-Item -ItemType Directory -Path "$repoRoot/logs" -Force | Out-Null
}

$PID | Out-File -FilePath $pidFile -Force -Encoding ascii

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Add-Content -Path $logFile -Value "[$ts] Starting Vite dev server on port 5173 (PID: $PID)..." -Encoding utf8

& npx.cmd vite --port 5173 --host 127.0.0.1 2>&1 | ForEach-Object {
    Add-Content -Path $logFile -Value $_ -Encoding utf8
}
