$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Stage D: train_v2 Status" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Alive / Dead Check
Write-Host "`n[PROCESS HEALTH]" -ForegroundColor Yellow
$brushProcs = Get-Process -Name "brush_app" -ErrorAction SilentlyContinue
$pidPath = "outputs/home/logs/train_v2.pid"
$psAlive = $false
if (Test-Path $pidPath) {
    $psPid = Get-Content $pidPath -ErrorAction SilentlyContinue
    if ($psPid) {
        $p = Get-Process -Id ([int]$psPid) -ErrorAction SilentlyContinue
        if ($p -and ($p.ProcessName -match "powershell")) {
            $psAlive = $true
        }
    }
}

$taskRunning = $false
try {
    $tq = schtasks /query /tn "home_train_v2" /fo csv 2>$null | ConvertFrom-Csv -ErrorAction SilentlyContinue
    if ($tq -and ($tq.Status -match "Running")) {
        $taskRunning = $true
    }
} catch {}

$donePath = "outputs/home/logs/train_v2.done"
if ($brushProcs -or $psAlive -or $taskRunning) {
    Write-Host "  Status: [ALIVE]" -ForegroundColor Green
    if ($taskRunning) { Write-Host "    - Scheduled Task: 'home_train_v2' is RUNNING" -ForegroundColor Green }
    if ($psAlive) { Write-Host "    - PowerShell Runner: PID $psPid is ALIVE" -ForegroundColor Green }
    if ($brushProcs) {
        foreach ($bp in $brushProcs) {
            $wsMB = [math]::Round($bp.WorkingSet64 / 1MB, 1)
            Write-Host "    - Brush: ID=$($bp.Id), CPU=$($bp.CPU)s, Mem=${wsMB}MB" -ForegroundColor Green
        }
    }
} elseif (Test-Path $donePath) {
    $doneTime = (Get-Item $donePath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "  Status: [COMPLETED] at $doneTime" -ForegroundColor Green
} else {
    Write-Host "  Status: [NOT RUNNING / DEAD]" -ForegroundColor Red
}

# 2. Step and ETA
Write-Host "`n[STEP & ETA]" -ForegroundColor Yellow
$logPath = "outputs/home/logs/train_v2.log"
$totalSteps = 30000
$currentStep = 0
$elapsedSec = 0
$etaMin = "Unknown"

if (Test-Path $logPath) {
    $logItem = Get-Item $logPath
    $lwTime = $logItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "  Log file : $logPath (Last modified: $lwTime)" -ForegroundColor Cyan
    
    # Read last 30 lines to extract step
    $tailLines = Get-Content $logPath -Tail 30 -ErrorAction SilentlyContinue
    foreach ($line in $tailLines) {
        if ($line -match "step\s+(\d+)|iter\s+(\d+)|(\d+)/30000|\[(\d+)/30000\]") {
            $matchesFound = $Matches[1], $Matches[2], $Matches[3], $Matches[4] | Where-Object { $_ }
            if ($matchesFound) {
                $currentStep = [int]$matchesFound[0]
            }
        }
    }
    
    # Start time
    $firstLine = Get-Content $logPath -Head 5 -ErrorAction SilentlyContinue | Where-Object { $_ -match "\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]" } | Select-Object -First 1
    if ($firstLine -and ($firstLine -match "\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")) {
        $startTime = [datetime]::ParseExact($Matches[1], "yyyy-MM-dd HH:mm:ss", $null)
        $elapsedSec = ((Get-Date) - $startTime).TotalSeconds
        if ($currentStep -gt 0 -and $elapsedSec -gt 10) {
            $rate = $currentStep / $elapsedSec
            $remSteps = [math]::Max(0, $totalSteps - $currentStep)
            $remSec = $remSteps / $rate
            $etaMin = [math]::Round($remSec / 60, 1)
        }
    }

    Write-Host "  Current Step   : $currentStep / $totalSteps ($([math]::Round($currentStep / $totalSteps * 100, 1))%)" -ForegroundColor Yellow
    Write-Host "  Elapsed Time   : $([math]::Round($elapsedSec / 60, 1)) min"
    Write-Host "  Estimated ETA  : ~$etaMin minutes"
    
    Write-Host "`n  --- Recent Log Output ---" -ForegroundColor Gray
    Get-Content $logPath -Tail 6 | ForEach-Object { Write-Host "    $_" }
} else {
    Write-Host "  Log file $logPath not found." -ForegroundColor Gray
}

# 3. GPU Telemetry
Write-Host "`n[GPU TELEMETRY]" -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    if ($gpuInfo) {
        Write-Host "  $gpuInfo" -ForegroundColor Green
    }
} catch {
    Write-Host "  nvidia-smi not available." -ForegroundColor Red
}

Write-Host "==================================================" -ForegroundColor Cyan
