param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$scene = "home"
$outputV2Dir = "outputs/$scene/v2"
$logDir = "outputs/$scene/logs"
if (-not (Test-Path $outputV2Dir)) { New-Item -ItemType Directory -Path $outputV2Dir -Force | Out-Null }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$logFile = "$logDir/train_v2.log"
$doneFile = "$logDir/train_v2.done"
$pidFile = "$logDir/train_v2.pid"
$taskFile = "$logDir/train_v2.task"

# Record PID and task name (R2)
Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path "logs/train_v2.pid" -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "home_train_v2" -Encoding utf8
Set-Content -Path "logs/train_v2.task" -Value "home_train_v2" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Stage D (train_v2) is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
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
    Add-Content -Path "logs/train_v2.log" -Value $line -Encoding utf8
}

Log-Msg "=================================================="
Log-Msg "  STAGE D: 3D Gaussian Splatting Training (final_v2)"
Log-Msg "  Scene: $scene | 30k steps, SH deg 3, scale reg, 1920 res"
Log-Msg "=================================================="

$brushExe = "tools/brush/brush_app.exe"
if (-not (Test-Path $brushExe)) {
    Log-Msg "[ERROR] Brush binary not found at $brushExe"
    exit 1
}

$trainPack = "outputs/$scene/train_pack"
if (-not (Test-Path $trainPack)) {
    Log-Msg "[ERROR] Training package not found at $trainPack"
    exit 1
}

Log-Msg "STARTING brush_app.exe training..."
Log-Msg "Settings: 30000 steps, max_splats=3000000, sh_degree=3, max_resolution=1920, scale_loss_weight=1e-7, opac_loss_weight=1e-8, eval_split_every=8"

$brushArgs = @(
    $trainPack,
    "--total-steps", "30000",
    "--max-splats", "3000000",
    "--sh-degree", "3",
    "--max-resolution", "1920",
    "--eval-split-every", "8",
    "--eval-every", "1000",
    "--eval-save-to-disk",
    "--scale-loss-weight", "1e-7",
    "--opac-loss-weight", "1e-8",
    "--export-every", "30000",
    "--export-path", $outputV2Dir,
    "--export-name", "scene_raw.ply"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Native -Exe $brushExe -CmdArgs $brushArgs -Log $logFile
$sw.Stop()

Log-Msg "FINISHED brush_app.exe in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes."

# Check exported raw ply
$exportedPly = "$outputV2Dir/scene_raw.ply"
if (-not (Test-Path $exportedPly)) {
    $alt = Get-ChildItem -Path $outputV2Dir -Filter "*.ply" | Select-Object -First 1
    if ($alt) {
        Copy-Item -Path $alt.FullName -Destination $exportedPly -Force
        Log-Msg "Copied $($alt.Name) to $exportedPly"
    }
}

if (Test-Path $exportedPly) {
    $sizeMB = [math]::Round((Get-Item $exportedPly).Length / 1MB, 2)
    Log-Msg "Exported scene_raw.ply: ${sizeMB} MB"
} else {
    Log-Msg "[WARN] scene_raw.ply not found in $outputV2Dir"
}

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
Set-Content -Path "logs/train_v2.done" -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8

try {
    schtasks /delete /tn home_train_v2 /f 2>&1 | Out-Null
    Log-Msg "Scheduled task 'home_train_v2' removed."
} catch {}

if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
if (Test-Path "logs/train_v2.pid") { Remove-Item -Force "logs/train_v2.pid" }
Log-Msg "[SUCCESS] Stage D complete and marked done."
