param(
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot/common.ps1"

if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }

$logFile = "logs/deploy_r2.log"
$doneFile = "logs/deploy_r2.done"
$pidFile = "logs/deploy_r2.pid"
$taskFile = "logs/deploy_r2.task"

Set-Content -Path $pidFile -Value $PID -Encoding utf8
Set-Content -Path $taskFile -Value "deploy_r2" -Encoding utf8

if ((Test-Path $doneFile) -and (-not $Force)) {
    Write-Host "[SKIP] R2 deploy already completed ($doneFile exists)."
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
Log-Msg "  STAGE 3: Cloudflare R2 Bucket & Asset Upload"
Log-Msg "=================================================="

$bucketName = "namma-ml-scene"

# 1. Create bucket (ignore error if it already exists)
Log-Msg "Creating R2 bucket '$bucketName'..."
try {
    Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "bucket", "create", $bucketName) $logFile
} catch {
    Log-Msg "Bucket creation returned: $_ (may already exist, continuing...)"
}

# 2. Enable r2.dev public access
Log-Msg "Enabling public access via r2.dev subdomain..."
try {
    Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "bucket", "dev-url", "enable", $bucketName) $logFile
} catch {
    Log-Msg "dev-url enable returned: $_ (may already be enabled, continuing...)"
}

# 3. Retrieve public dev-url
Log-Msg "Retrieving public r2.dev URL..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "bucket", "dev-url", "get", $bucketName) $logFile

# 4. Set CORS rules
Log-Msg "Setting CORS configuration on '$bucketName'..."
$corsFile = "docs/deploy/r2_cors.json"
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "bucket", "cors", "set", $bucketName, "--file", $corsFile) $logFile

# 5. Upload files to R2
Log-Msg "Uploading scene.v1.spz..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "object", "put", "$bucketName/scene.v1.spz", "--file", "viewer/public/scene/scene.v1.spz", "--content-type", "application/octet-stream", "--cache-control", "public, max-age=31536000, immutable", "--remote") $logFile

Log-Msg "Uploading occupancy.json..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "object", "put", "$bucketName/occupancy.json", "--file", "viewer/public/scene/occupancy.json", "--content-type", "application/json", "--cache-control", "public, max-age=31536000, immutable", "--remote") $logFile

Log-Msg "Uploading occupancy.png..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "object", "put", "$bucketName/occupancy.png", "--file", "viewer/public/scene/occupancy.png", "--content-type", "image/png", "--cache-control", "public, max-age=31536000, immutable", "--remote") $logFile

Log-Msg "Uploading scene.json..."
Invoke-Native "cmd.exe" @("/c", "npx", "wrangler", "r2", "object", "put", "$bucketName/scene.json", "--file", "viewer/public/scene/scene.json", "--content-type", "application/json", "--cache-control", "public, max-age=60", "--remote") $logFile

Set-Content -Path $doneFile -Value "DONE" -Encoding utf8
Log-Msg "[SUCCESS] R2 bucket setup and asset upload complete."

