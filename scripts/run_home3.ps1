param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

$logDir = "data/home3/logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$masterLog = "$logDir/run_home3.log"
$pidFile = "$logDir/run_home3.pid"
$taskFile = "$logDir/run_home3.task"
$doneFile = "$logDir/home3.done"

# Record PID and task name (R2)
Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path "logs/home3.pid" -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "home3_sfm" -Encoding utf8
Set-Content -Path "logs/home3.task" -Value "home3_sfm" -Encoding utf8

function Log-Msg {
    param([string]$msg)
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $masterLog -Value $line -Encoding utf8
    Add-Content -Path "logs/home3.log" -Value $line -Encoding utf8
}

Log-Msg "=================================================="
Log-Msg "  Namma Space: home3 Reconstruction Pipeline"
Log-Msg "  Extending working home reconstruction with sweeps & stills"
Log-Msg "=================================================="

$stages = @(
    @{ Name = "features"; Script = "scripts/home3_features.ps1" },
    @{ Name = "matching"; Script = "scripts/home3_matching.ps1" },
    @{ Name = "mapping";  Script = "scripts/home3_mapping.ps1" }
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($stage in $stages) {
    $stageDone = "data/home3/logs/$($stage.Name).done"
    if ((Test-Path $stageDone) -and (-not $Force)) {
        Log-Msg "[SKIPPING] Stage '$($stage.Name)' is already completed ($stageDone exists)."
        continue
    }

    Log-Msg ">>> [EXECUTING] Stage '$($stage.Name)': $($stage.Script)"
    $stageSw = [System.Diagnostics.Stopwatch]::StartNew()

    $cmdArgs = if ($Force) { @("-ExecutionPolicy", "Bypass", "-File", $stage.Script, "-Force") } else { @("-ExecutionPolicy", "Bypass", "-File", $stage.Script) }
    
    Invoke-Native -Exe "powershell" -CmdArgs $cmdArgs -Log $masterLog

    $stageSw.Stop()
    Log-Msg ">>> [COMPLETED] Stage '$($stage.Name)' in $([math]::Round($stageSw.Elapsed.TotalSeconds, 1))s."
}

$sw.Stop()
Log-Msg "=================================================="
Log-Msg "  home3 PIPELINE COMPLETE in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes!"
Log-Msg "  Sparse extended model output: data/home3/colmap/sparse_ext/"
Log-Msg "  Analysis report             : data/home3/logs/analysis.txt"
Log-Msg "=================================================="

Set-Content -Path $doneFile -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8
Set-Content -Path "logs/home3.done" -Value "DONE at $(Get-Date -Format 'o')" -Encoding utf8

# Clean up task (R2)
try {
    schtasks /delete /tn home3_sfm /f 2>&1 | Out-Null
    Log-Msg "Scheduled task 'home3_sfm' removed."
} catch {
    Log-Msg "Note: Could not delete scheduled task 'home3_sfm'."
}

if (Test-Path $pidFile) { Remove-Item -Force $pidFile }
if (Test-Path "logs/home3.pid") { Remove-Item -Force "logs/home3.pid" }
