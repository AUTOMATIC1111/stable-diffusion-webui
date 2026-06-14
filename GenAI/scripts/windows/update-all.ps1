#Requires -Version 5.1
<#
.SYNOPSIS
  Interactive upstream update helper. Does not auto-upgrade major versions.
#>
param([switch]$Force)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$cfg = Get-GenAIConfig
$lockPath = Join-Path $cfg.Root 'upstreams.lock.json'
$log = Join-Path (Resolve-GenAIPath $cfg.Root 'logs') 'update.log'
Ensure-GenAIDirectories $cfg

Write-Host 'GenAI Update — pinned upstream management'
Write-Host "Current ComfyUI:    $($cfg.Lock.upstreams.comfyui.release) @ $($cfg.Lock.upstreams.comfyui.commit)"
Write-Host "Current FaceFusion: $($cfg.Lock.upstreams.facefusion.release) @ $($cfg.Lock.upstreams.facefusion.commit)"
Write-Host ''
Write-Host 'This project pins explicit release commits. To update:'
Write-Host '  1. Review upstream release notes on GitHub'
Write-Host '  2. Edit upstreams.lock.json with new release tag and commit SHA'
Write-Host '  3. Run: python tests/validate-upstreams.py'
Write-Host '  4. Run: .\scripts\windows\setup-all.ps1'
Write-Host '  5. Run: .\scripts\windows\doctor.ps1'
Write-Host ''
Write-Host 'Rollback: restore upstreams.lock.json from Git and re-run setup.'

Write-GenAILog $log 'Update check displayed pinned versions (no automatic upgrade)'
