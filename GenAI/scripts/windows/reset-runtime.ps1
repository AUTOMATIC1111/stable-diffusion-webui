#Requires -Version 5.1
<#
.SYNOPSIS
  Destructive reset of runtime clones and virtual environments only.
#>
param([switch]$Confirm)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

if (-not $Confirm) {
    Write-Host 'This removes runtime clones and venvs. Models, inputs, and outputs are preserved.'
    Write-Host 'Re-run with -Confirm to proceed.'
    exit 1
}

$cfg = Get-GenAIConfig
$targets = @(
    $cfg.Paths.comfyui.cloneDir,
    $cfg.Paths.facefusion.cloneDir,
    $cfg.Paths.comfyui.envDir,
    $cfg.Paths.facefusion.envDir,
    $cfg.Paths.facefusion.tempDir
)

foreach ($rel in $targets) {
    $full = Resolve-GenAIPath $cfg.Root $rel
    if (Test-Path $full) {
        Remove-Item $full -Recurse -Force
        Write-Host "Removed $full"
    }
}
Write-Host 'Reset complete. Run setup-all.ps1 to rebuild.'
