#Requires -Version 5.1
param(
    [ValidateSet('compatibility', 'balanced', 'performance')]
    [string]$Profile = 'balanced'
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\setup-comfyui.ps1" -Profile $Profile
& "$PSScriptRoot\setup-facefusion.ps1"
Write-Host 'GenAI setup-all complete. Run .\scripts\windows\doctor.ps1 to verify.'
