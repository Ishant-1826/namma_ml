param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$outputMaxDir = "outputs/home3/max"
$logDir = "outputs/home3/logs"
if (-not (Test-Path $outputMaxDir)) { New-Item -ItemType Directory -Path $outputMaxDir -Force | Out-Null }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$logFile = "$logDir/train_max.log"
$doneFile = "$logDir/train_max.done"
$pidFile = "$logDir/train_max.pid"
$taskFile = "$logDir/train_max.task"

# Record PID and task name (R2)
Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path "logs/train_home3_max.pid" -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "train_home3_max" -Encoding utf8
Set-Content -Path "logs/train_home3_max.task" -Value "train_home3_max" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Stage R2 (train_home3_max) is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
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
    Add-Content -Path "logs/train_home3_max.log" -Value $line -Encoding utf8
}

Log-Msg "=================================================="
Log-Msg "  STAGE R2: 3D Gaussian Splatting Training (home3_max)"
Log-Msg "  Dataset: outputs/home3/train_pack (634 frames)"
Log-Msg "  Settings: 30k steps, sh_degree=0 (LOOK-BACK FIX), scale_loss_weight=1e-7, res=1920"
Log-Msg "=================================================="

$brushExe = "tools/brush/brush_app.exe"
if (-not (Test-Path $brushExe)) {
    Log-Msg "[ERROR] Brush binary not found at $brushExe"
    exit 1
}

$trainPack = "outputs/home3/train_pack"
if (-not (Test-Path $trainPack)) {
    Log-Msg "[ERROR] Training package not found at $trainPack"
    exit 1
}

Log-Msg "STARTING brush_app.exe training..."

$brushArgs = @(
    $trainPack,
    "--total-steps", "30000",
    "--max-splats", "3000000",
    "--sh-degree", "0",
    "--max-resolution", "1920",
    "--eval-split-every", "8",
    "--eval-every", "2000",
    "--eval-save-to-disk",
    "--scale-loss-weight", "1e-7",
    "--opac-loss-weight", "1e-8",
    "--export-every", "30000",
    "--export-path", $outputMaxDir,
    "--export-name", "scene_raw.ply"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Native -Exe $brushExe -CmdArgs $brushArgs -Log $logFile
$sw.Stop()

Log-Msg "FINISHED brush_app.exe in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes."

# Check exported raw ply
$exportedPly = "$outputMaxDir/scene_raw.ply"
if (-not (Test-Path $exportedPly)) {
    $alt = Get-ChildItem -Path $outputMaxDir -Filter "*.ply" | Select-Object -First 1
    if ($alt) {
        Copy-Item -Path $alt.FullName -Destination $exportedPly -Force
        Log-Msg "Copied $($alt.Name) to $exportedPly"
    }
}

if (Test-Path $exportedPly) {
    $sizeMB = [math]::Round((Get-Item $exportedPly).Length / 1MB, 2)
    Log-Msg "Exported scene_raw.ply: ${sizeMB} MB"
} else {
    Log-Msg "[WARN] scene_raw.ply not found in $outputMaxDir"
}

# Run Stage R3 postprocessing on the new raw ply
Log-Msg "Running pipeline/postprocess_max.py on home3_max raw splats..."
$postArgs = @(
    "pipeline/postprocess_max.py",
    "--raw-ply", $exportedPly,
    "--output-dir", $outputMaxDir,
    "--transforms", "$trainPack/transforms.json",
    "--scene", "home"
)
Invoke-Native -Exe "python" -CmdArgs $postArgs -Log $logFile

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
Set-Content -Path "logs/train_home3_max.done" -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8

try {
    schtasks /delete /tn train_home3_max /f 2>&1 | Out-Null
    Log-Msg "Scheduled task 'train_home3_max' removed."
} catch {}

if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
if (Test-Path "logs/train_home3_max.pid") { Remove-Item -Force "logs/train_home3_max.pid" }
Log-Msg "[SUCCESS] Stage R2 complete and marked done."
