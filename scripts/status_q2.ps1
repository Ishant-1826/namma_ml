$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Q2: Floater Removal Status" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$donePath = "$repoRoot/logs/q2_splats.done"
$logPath = "$repoRoot/logs/q2_splats.log"
$pidPath = "$repoRoot/logs/q2_splats.pid"

$taskRunning = $false
try {
    $tq = schtasks /query /tn "home_q2_splats" /fo csv 2>$null | ConvertFrom-Csv -ErrorAction SilentlyContinue
    if ($tq -and ($tq.Status -match "Running")) {
        $taskRunning = $true
    }
} catch {}

$psAlive = $false
if (Test-Path $pidPath) {
    $psPid = Get-Content $pidPath -ErrorAction SilentlyContinue
    if ($psPid) {
        $p = Get-Process -Id ([int]$psPid) -ErrorAction SilentlyContinue
        if ($p) { $psAlive = $true }
    }
}

$pyProc = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    try { $_.CommandLine -match "build_q2_splats" } catch { $true }
}

if ($taskRunning -or $psAlive -or $pyProc) {
    Write-Host "  Status: [RUNNING]" -ForegroundColor Yellow
} elseif (Test-Path $donePath) {
    Write-Host "  Status: [COMPLETED]" -ForegroundColor Green
} else {
    Write-Host "  Status: [NOT RUNNING / FAILED]" -ForegroundColor Red
}

if (Test-Path $logPath) {
    Write-Host "`nLast 10 log lines:" -ForegroundColor Gray
    Get-Content $logPath -Tail 10
}
