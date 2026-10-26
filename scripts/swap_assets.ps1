param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Folder
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$targetDir = "viewer/public/scene"
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

if (-not (Test-Path $Folder)) {
    Write-Error "Source folder '$Folder' does not exist."
    exit 1
}

$filesToCopy = @("scene.json", "scene.spz", "scene.splat", "occupancy.json", "occupancy.png")
Write-Host "Swapping assets from '$Folder' into '$targetDir'..." -ForegroundColor Cyan

$copiedCount = 0
foreach ($f in $filesToCopy) {
    $srcPath = Join-Path $Folder $f
    if (Test-Path $srcPath) {
        Copy-Item -Path $srcPath -Destination $targetDir -Force
        $size = (Get-Item $srcPath).Length
        $sizeStr = if ($size -ge 1MB) { "$([math]::Round($size / 1MB, 2)) MB" } else { "$([math]::Round($size / 1KB, 1)) KB" }
        Write-Host "  [OK] Copied $f ($sizeStr)" -ForegroundColor Green
        $copiedCount++
    } else {
        Write-Host "  [WARN] File $f not found in $Folder, skipped." -ForegroundColor Yellow
    }
}

Write-Host "Successfully swapped $copiedCount assets into $targetDir." -ForegroundColor Green
