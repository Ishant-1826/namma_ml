param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$logDir = "$repoRoot/logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$logFile = "$logDir/q2_splats.log"
$doneFile = "$logDir/q2_splats.done"
$pidFile = "$logDir/q2_splats.pid"
$taskFile = "$logDir/q2_splats.task"

Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "home_q2_splats" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Q2 is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
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
Log-Msg "  STAGE Q2: Floater Removal from Free Space"
Log-Msg "=================================================="

$pyExe = "$repoRoot/.venv/Scripts/python.exe"
if (-not (Test-Path $pyExe)) {
    $pyExe = "python"
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Native -Exe $pyExe -CmdArgs @("scripts/build_q2_splats.py") -Log $logFile
$sw.Stop()

Log-Msg "FINISHED Q2 in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes."

# Verify outputs
$outputCleanPly = "$repoRoot/outputs/home/v3/scene_clean.ply"
$outputSplat = "$repoRoot/outputs/home/v3/scene.splat"
$outputSpz = "$repoRoot/outputs/home/v3/scene.spz"

if ((Test-Path $outputCleanPly) -and (Test-Path $outputSplat) -and (Test-Path $outputSpz)) {
    Set-Content -Path $doneFile -Value "SUCCESS" -Encoding utf8
    Log-Msg "[SUCCESS] Q2 completed successfully. Done file written: $doneFile"
} else {
    Log-Msg "[ERROR] One or more Q2 output files missing."
    exit 1
}
