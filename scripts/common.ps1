# scripts/common.ps1
# Standing helper for native command execution in PowerShell (Rule R4)

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$Exe,

        [Parameter(Mandatory = $true, Position = 1)]
        [string[]]$CmdArgs,

        [Parameter(Mandatory = $true, Position = 2)]
        [string]$Log
    )

    $origPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $logDir = Split-Path -Parent $Log
    if ($logDir -and (-not (Test-Path $logDir))) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $Log -Value "[$ts] [INVOKE-NATIVE] $Exe $($CmdArgs -join ' ')" -Encoding utf8

    try {
        & $Exe @CmdArgs 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Add-Content -Path $Log -Value $line -Encoding utf8
            Write-Output $line
        }
    }
    finally {
        $ErrorActionPreference = $origPref
    }

    if ($LASTEXITCODE -ne 0) {
        $tailLines = Get-Content -Path $Log -Tail 20 -ErrorAction SilentlyContinue
        Write-Host "================== LAST 20 LOG LINES ($Log) ==================" -ForegroundColor Red
        if ($tailLines) {
            foreach ($l in $tailLines) { Write-Host $l -ForegroundColor Red }
        }
        Write-Host "==============================================================" -ForegroundColor Red
        throw "Command '$Exe' exited with non-zero exit code: $LASTEXITCODE"
    }
}
