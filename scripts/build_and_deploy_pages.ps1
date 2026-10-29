param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$logFile = "logs/deploy_pages.log"
$doneFile = "logs/deploy_pages.done"
$pidFile = "logs/deploy_pages.pid"
$taskFile = "logs/deploy_pages.task"

Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "deploy_pages" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Pages deploy already completed ($doneFile exists)."
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
Log-Msg "  STAGE 5: Build Viewer & Deploy to Cloudflare Pages"
Log-Msg "=================================================="

# 1. Run tests first to ensure zero regressions
Log-Msg "Running vitest in viewer/..."
Set-Location "$repoRoot/viewer"
Invoke-Native "cmd.exe" @("/c", "npm", "test") "$repoRoot/$logFile"

# 2. Run npm run build
Log-Msg "Running npm run build in viewer/..."
Invoke-Native "cmd.exe" @("/c", "npm", "run", "build") "$repoRoot/$logFile"
Set-Location $repoRoot

# 3. Audit all files in viewer/dist
Log-Msg "Auditing viewer/dist sizes for Pages 25 MiB per-file limit..."
$maxSizeBytes = 25 * 1024 * 1024
$files = Get-ChildItem -Path "viewer/dist" -Recurse -File
$oversized = @()

foreach ($f in $files) {
    $sizeMB = ($f.Length / 1MB).ToString("0.00")
    Log-Msg "  File: $($f.FullName.Replace($repoRoot, '')) - $sizeMB MB ($($f.Length) bytes)"
    if ($f.Length -gt $maxSizeBytes) {
        $oversized += $f.FullName
    }
}

if ($oversized.Count -gt 0) {
    Log-Msg "[ERROR] Found files exceeding Pages 25 MiB limit: $($oversized -join ', ')"
    exit 1
}
Log-Msg "[PASS] All files in viewer/dist are under 25 MiB."

# 4. Deploy to Cloudflare Pages
Log-Msg "Deploying viewer/dist to Cloudflare Pages project 'namma-ml-viewer'..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "pages", "deploy", "viewer/dist", "--project-name", "namma-ml-viewer", "--branch", "main") $logFile

Set-Content -Path $doneFile -Value "DONE" -Encoding utf8
Log-Msg "[SUCCESS] Cloudflare Pages deployment completed successfully."

