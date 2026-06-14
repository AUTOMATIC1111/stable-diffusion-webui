#Requires -Version 5.1
param(
    [ValidateSet('compatibility', 'balanced', 'performance')]
    [string]$Profile = ''
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$cfg = Get-GenAIConfig
Ensure-GenAIDirectories $cfg

if (-not $Profile) {
    $Profile = Get-SettingOrDefault $cfg.Settings 'GENAI_PROFILE' 'balanced'
}

$root = $cfg.Root
$cloneDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.cloneDir
$envDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.envDir
$inputDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.inputDir
$outputDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.outputDir
$python = Join-Path $envDir 'Scripts\python.exe'
$hostAddr = Get-SettingOrDefault $cfg.Settings 'GENAI_HOST' '127.0.0.1'
$port = Get-SettingOrDefault $cfg.Settings 'COMFYUI_PORT' '8188'

if (-not (Test-Path $python)) {
    throw "ComfyUI environment missing. Run: .\scripts\windows\setup-comfyui.ps1"
}

$launchArgs = @(
    (Join-Path $cloneDir 'main.py'),
    '--listen', $hostAddr,
    '--port', $port,
    '--input-directory', $inputDir,
    '--output-directory', $outputDir
)

switch ($Profile) {
    'compatibility' { $launchArgs += @('--disable-smart-memory') }
    'performance'   { $launchArgs += @('--highvram') }
}

if ($hostAddr -notin @('127.0.0.1', 'localhost')) {
    Write-Warning "ComfyUI binding to $hostAddr — ensure this is intentional. Default is localhost-only."
}

$cudaInfo = & $python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>$null

Write-Host '=== ComfyUI Launch ==='
Write-Host "Version:   $($cfg.Lock.upstreams.comfyui.release) ($($cfg.Lock.upstreams.comfyui.commit.Substring(0,7)))"
Write-Host "Profile:   $Profile"
Write-Host "Env:       $envDir"
Write-Host "Backend:   $cudaInfo"
Write-Host "Address:   http://${hostAddr}:${port}"
Write-Host "Input:     $inputDir"
Write-Host "Output:    $outputDir"
Write-Host "Models:    $(Resolve-GenAIPath $root $cfg.Paths.comfyui.modelsDir)"
Write-Host "Logs:      $(Resolve-GenAIPath $root $cfg.Paths.comfyui.logDir)"
Write-Host '======================'

Push-Location $cloneDir
try {
    & $python @launchArgs
} finally {
    Pop-Location
}
