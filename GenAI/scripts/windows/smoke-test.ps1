#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $root
try {
    python tests\validate-config.py
    python tests\validate-upstreams.py
    python tests\validate-workflows.py
    python tests\smoke_test.py --static-only
    Write-Host 'Smoke tests passed (static validation).'
} finally {
    Pop-Location
}
