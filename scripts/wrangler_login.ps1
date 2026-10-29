param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$logFile = "logs/wrangler_login.log"
$doneFile = "logs/wrangler_login.done"
$pidFile = "logs/wrangler_login.pid"
$taskFile = "logs/wrangler_login.task"

Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "wrangler_login" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Already logged in ($doneFile exists)."
    exit 0
}

if ($Force -and (Test-Path $doneFile)) {
    Remove-Item -Force $doneFile
}

function Log-Msg {
    param([string]$msg)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Log-Msg "=================================================="
Log-Msg "  Starting Cloudflare Wrangler Login"
Log-Msg "=================================================="

# Run wrangler login
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "login") $logFile

# Verify authentication with wrangler whoami
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "whoami") $logFile

Set-Content -Path $doneFile -Value "DONE" -Encoding utf8
Log-Msg "[SUCCESS] Wrangler login complete."

