param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$logDir = "data/home3/logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$doneFile = "$logDir/features.done"
$logFile = "$logDir/features.log"
$pidFile = "$logDir/features.pid"

Set-Content -Path $pidFile -Value $PID -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Stage 'features' is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
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
Log-Msg "  STAGE: home3 Feature Extraction (GPU SIFT)"
Log-Msg "=================================================="

# Count images in DB before
$countBefore = python -c "import sqlite3; conn = sqlite3.connect('data/home3/colmap/database.db'); cur = conn.cursor(); cur.execute('SELECT count(*) FROM images'); print(cur.fetchone()[0]); conn.close()"
Log-Msg "Images in database before extraction: $countBefore"

Log-Msg "STARTING colmap feature_extractor on new images..."
Log-Msg "Settings: model=SIMPLE_RADIAL, single_camera_per_folder=1, focal_factor=0.75, prior=1440,960,540,0, use_gpu=1"

$colmapArgs = @(
    "feature_extractor",
    "--database_path", "data/home3/colmap/database.db",
    "--image_path", "data/home3/images",
    "--image_list_path", "data/home3/new_images.txt",
    "--ImageReader.camera_model", "SIMPLE_RADIAL",
    "--ImageReader.single_camera_per_folder", "1",
    "--ImageReader.default_focal_length_factor", "0.75",
    "--ImageReader.camera_params", "1440,960,540,0",
    "--SiftExtraction.use_gpu", "1",
    "--SiftExtraction.max_num_features", "8192"
)

Invoke-Native -Exe "colmap" -CmdArgs $colmapArgs -Log $logFile

Log-Msg "FINISHED colmap feature_extractor."

# Count images in DB after
$countAfter = python -c "import sqlite3; conn = sqlite3.connect('data/home3/colmap/database.db'); cur = conn.cursor(); cur.execute('SELECT count(*) FROM images'); print(cur.fetchone()[0]); conn.close()"
$added = [int]$countAfter - [int]$countBefore
Log-Msg "Images in database after extraction: $countAfter (added $added new images)"

# Check cameras count in DB
$camsAfter = python -c "import sqlite3; conn = sqlite3.connect('data/home3/colmap/database.db'); cur = conn.cursor(); cur.execute('SELECT count(*) FROM cameras'); print(cur.fetchone()[0]); conn.close()"
Log-Msg "Cameras in database: $camsAfter"

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
Log-Msg "[SUCCESS] Stage 'features' completed and marked done ($doneFile)."
