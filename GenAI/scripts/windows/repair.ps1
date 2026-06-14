#Requires -Version 5.1
<#
.SYNOPSIS
  Repair environments without deleting models or outputs.
#>
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$cfg = Get-GenAIConfig
$log = Join-Path (Resolve-GenAIPath $cfg.Root 'logs') 'repair.log'
Write-GenAILog $log 'Starting repair (preserving models and outputs)'

& "$PSScriptRoot\setup-comfyui.ps1"
& "$PSScriptRoot\setup-facefusion.ps1"

Write-GenAILog $log 'Repair complete'
Write-Host 'Repair finished. Run doctor.ps1 to verify.'
