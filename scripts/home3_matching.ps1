param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$logDir = "data/home3/logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$doneFile = "$logDir/matching.done"
$logFile = "$logDir/matching.log"
$pidFile = "$logDir/matching.pid"

Set-Content -Path $pidFile -Value $PID -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] Stage 'matching' is already completed ($doneFile exists). Use -Force to re-run." -ForegroundColor Green
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
Log-Msg "  STAGE: home3 Pair Generation & Matching"
Log-Msg "=================================================="

# 1. Run make_home3_pairs.py
Log-Msg "Running scripts/make_home3_pairs.py to generate data/home3/pairs.txt..."
python scripts/make_home3_pairs.py 2>&1 | Add-Content -Path $logFile -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Log-Msg "[ERROR] scripts/make_home3_pairs.py failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$pairsFile = "data/home3/pairs.txt"
if (-not (Test-Path $pairsFile)) {
    Log-Msg "[ERROR] Pairs file $pairsFile was not generated."
    exit 1
}

$pairLines = (Get-Content $pairsFile | Measure-Object -Line).Lines
Log-Msg "Total candidate pairs in ${pairsFile}: $pairLines"

# 2. Run colmap matches_importer using Invoke-Native
Log-Msg "STARTING colmap matches_importer on $pairLines pairs..."
Log-Msg "Settings: match_type=pairs, use_gpu=1, guided_matching=1, max_ratio=0.80"

$colmapArgs = @(
    "matches_importer",
    "--database_path", "data/home3/colmap/database.db",
    "--match_list_path", "data/home3/pairs.txt",
    "--match_type", "pairs",
    "--SiftMatching.use_gpu", "1",
    "--SiftMatching.guided_matching", "1",
    "--SiftMatching.max_ratio", "0.80"
)

Invoke-Native -Exe "colmap" -CmdArgs $colmapArgs -Log $logFile

Log-Msg "FINISHED colmap matches_importer."

$finalPairsInDb = python -c "import sqlite3; conn = sqlite3.connect('data/home3/colmap/database.db'); cur = conn.cursor(); cur.execute('SELECT count(*) FROM two_view_geometries WHERE rows > 0'); print(cur.fetchone()[0]); conn.close()"
Log-Msg "Total verified two-view geometries in database: $finalPairsInDb"

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
Log-Msg "[SUCCESS] Stage 'matching' completed and marked done ($doneFile)."
