param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$logDir = "data/home3/logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$doneFile = "$logDir/mapping.done"
$logFile = "$logDir/mapping.log"
$pidFile = "$logDir/mapping.pid"
$analysisFile = "$logDir/analysis.txt"

Set-Content -Path $pidFile -Value $PID -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Stage 'mapping' is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
    exit 0
}

if ($Force) {
    if (Test-Path $doneFile) { Remove-Item -Force $doneFile }
    if (Test-Path "data/home3/colmap/sparse_ext") { Remove-Item -Recurse -Force "data/home3/colmap/sparse_ext" }
}

function Log-Msg {
    param([string]$msg)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Log-Msg "=================================================="
Log-Msg "  STAGE: home3 Mapper Continuation"
Log-Msg "=================================================="

$inputModel = "data/home3/colmap/sparse/0"
$outputModelDir = "data/home3/colmap/sparse_ext"

if (-not (Test-Path $inputModel)) {
    Log-Msg "[ERROR] Input model not found at $inputModel"
    exit 1
}

if (-not (Test-Path $outputModelDir)) {
    New-Item -ItemType Directory -Path $outputModelDir -Force | Out-Null
}

Log-Msg "Input model: $inputModel"
Log-Msg "Output dir : $outputModelDir"
Log-Msg "Settings: --Mapper.fix_existing_images 1 (original 446 keep poses), ba_refine_focal_length=1, ba_refine_principal_point=0, ba_refine_extra_params=0"

Log-Msg "STARTING colmap mapper continuation..."

$colmapArgs = @(
    "mapper",
    "--database_path", "data/home3/colmap/database.db",
    "--image_path", "data/home3/images",
    "--input_path", $inputModel,
    "--output_path", $outputModelDir,
    "--Mapper.fix_existing_images", "1",
    "--Mapper.ba_refine_focal_length", "1",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0",
    "--Mapper.abs_pose_min_num_inliers", "20",
    "--Mapper.abs_pose_min_inlier_ratio", "0.20",
    "--Mapper.max_reg_trials", "3",
    "--Mapper.init_min_tri_angle", "8",
    "--Mapper.init_min_num_inliers", "50",
    "--Mapper.filter_max_reproj_error", "4.0",
    "--Mapper.filter_min_tri_angle", "1.5",
    "--Mapper.multiple_models", "0"
)

Invoke-Native -Exe "colmap" -CmdArgs $colmapArgs -Log $logFile

Log-Msg "FINISHED colmap mapper."

# Locate the resulting model directory (colmap usually puts it in sparse_ext/0)
$analyzedPath = if (Test-Path "$outputModelDir/0/cameras.bin") {
    "$outputModelDir/0"
} elseif (Test-Path "$outputModelDir/0/cameras.txt") {
    "$outputModelDir/0"
} elseif (Test-Path "$outputModelDir/cameras.bin") {
    "$outputModelDir"
} elseif (Test-Path "$outputModelDir/cameras.txt") {
    "$outputModelDir"
} else {
    $sub = Get-ChildItem -Path $outputModelDir -Directory | Select-Object -First 1
    if ($sub) { $sub.FullName } else { $outputModelDir }
}

Log-Msg "Running colmap model_analyzer on $analyzedPath..."
$analyzerArgs = @(
    "model_analyzer",
    "--path", $analyzedPath
)

Invoke-Native -Exe "colmap" -CmdArgs $analyzerArgs -Log $analysisFile

if (Test-Path $analysisFile) {
    Get-Content $analysisFile | Add-Content -Path $logFile -Encoding utf8
}

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
Log-Msg "[SUCCESS] Stage 'mapping' completed and marked done ($doneFile)."
Log-Msg "Analysis saved to $analysisFile."
