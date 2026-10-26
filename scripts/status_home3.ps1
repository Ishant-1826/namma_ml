$ErrorActionPreference = 'SilentlyContinue'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  home3 Pipeline Status" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Pipeline Alive or Dead Check
Write-Host "`n[OVERALL PIPELINE HEALTH]" -ForegroundColor Yellow
$colmapProcs = Get-Process -Name "colmap" -ErrorAction SilentlyContinue
$pidPath = "data/home3/logs/run_home3.pid"
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
    $taskQuery = schtasks /query /tn "home3_sfm" /fo csv 2>$null | ConvertFrom-Csv -ErrorAction SilentlyContinue
    if ($taskQuery -and ($taskQuery.Status -match "Running")) {
        $taskRunning = $true
    }
} catch {}

if ($colmapProcs -or $psAlive -or $taskRunning) {
    Write-Host "  Pipeline Status: [ALIVE] (Active execution detected)" -ForegroundColor Green
    if ($taskRunning) { Write-Host "    - Scheduled Task: 'home3_sfm' is RUNNING" -ForegroundColor Green }
    if ($psAlive) { Write-Host "    - Runner PowerShell: PID $psPid is ALIVE" -ForegroundColor Green }
    if ($colmapProcs) {
        foreach ($cp in $colmapProcs) {
            $wsMB = [math]::Round($cp.WorkingSet64 / 1MB, 1)
            Write-Host "    - COLMAP: ID=$($cp.Id), Name=$($cp.ProcessName), CPU=$($cp.CPU)s, Mem=${wsMB}MB" -ForegroundColor Green
        }
    }
} else {
    $allDone = (Test-Path "data/home3/logs/home3.done")
    if ($allDone) {
        Write-Host "  Pipeline Status: [COMPLETED]" -ForegroundColor Green
    } else {
        Write-Host "  Pipeline Status: [NOT RUNNING / DEAD]" -ForegroundColor Red
    }
}

# 2. Stage completion status
Write-Host "`n[STAGES STATUS]" -ForegroundColor Yellow
$stages = @("features", "matching", "mapping")
foreach ($st in $stages) {
    $donePath = "data/home3/logs/$st.done"
    if (Test-Path $donePath) {
        $doneTime = (Get-Item $donePath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "  - Stage '$st': [COMPLETED] at $doneTime" -ForegroundColor Green
    } else {
        $logPath = "data/home3/logs/$st.log"
        if (Test-Path $logPath) {
            $logTime = (Get-Item $logPath).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            Write-Host "  - Stage '$st': [RUNNING/PENDING] (Log modified: $logTime)" -ForegroundColor Yellow
        } else {
            Write-Host "  - Stage '$st': [NOT STARTED]" -ForegroundColor Gray
        }
    }
}

# 3. Newest log tail (last 8 lines + last write time)
Write-Host "`n[NEWEST LOG TAIL]" -ForegroundColor Yellow
$logDir = "data/home3/logs"
if (Test-Path $logDir) {
    $newestLog = Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending |
                 Select-Object -First 1
    if ($newestLog) {
        $lwTime = $newestLog.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "  Log: $($newestLog.Name) (Last write: $lwTime)" -ForegroundColor Cyan
        Write-Host "  --- Last 8 lines ---" -ForegroundColor Gray
        Get-Content $newestLog.FullName -Tail 8 | ForEach-Object { Write-Host "    $_" }
    } else {
        Write-Host "  No log files found in $logDir." -ForegroundColor Gray
    }
} else {
    Write-Host "  Logs directory $logDir does not exist yet." -ForegroundColor Gray
}

# 4. Database size and last write time
Write-Host "`n[DATABASE STATUS]" -ForegroundColor Yellow
$dbPath = "data/home3/colmap/database.db"
if (Test-Path $dbPath) {
    $dbItem = Get-Item $dbPath
    $dbSizeMB = [math]::Round($dbItem.Length / 1MB, 2)
    $dbTime = $dbItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Host "  Database: $dbPath" -ForegroundColor Cyan
    Write-Host "  Size    : ${dbSizeMB} MB ($($dbItem.Length) bytes)"
    Write-Host "  Modified: $dbTime"
} else {
    Write-Host "  Database $dbPath not found." -ForegroundColor Red
}

# 5. NVIDIA-SMI GPU Use
Write-Host "`n[GPU TELEMETRY]" -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    if ($gpuInfo) {
        Write-Host "  $gpuInfo" -ForegroundColor Green
    } else {
        Write-Host "  nvidia-smi returned empty." -ForegroundColor Gray
    }
} catch {
    Write-Host "  nvidia-smi not available." -ForegroundColor Red
}

Write-Host "==================================================" -ForegroundColor Cyan
