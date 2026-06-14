#Requires -Version 5.1
param(
    [string]$ExecutionProvider = ''
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$cfg = Get-GenAIConfig
Ensure-GenAIDirectories $cfg

$root = $cfg.Root
$cloneDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.cloneDir
$envDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.envDir
$configPath = Join-Path $root 'config\facefusion.ini'
$python = Join-Path $envDir 'Scripts\python.exe'
$hostAddr = Get-SettingOrDefault $cfg.Settings 'GENAI_HOST' '127.0.0.1'
$port = Get-SettingOrDefault $cfg.Settings 'FACEFUSION_PORT' '7861'
$tempDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.tempDir
$outputDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.outputDir

if (-not $ExecutionProvider) {
    $ExecutionProvider = Get-SettingOrDefault $cfg.Settings 'FACEFUSION_EXECUTION_PROVIDER' 'cuda'
}

if (-not (Test-Path $python)) {
    throw "FaceFusion environment missing. Run: .\scripts\windows\setup-facefusion.ps1"
}

if (-not (Test-Path $configPath)) {
    Copy-Item (Join-Path $root 'config\facefusion.example.ini') $configPath
}

$providers = switch ($ExecutionProvider) {
    'cuda' { 'cuda' }
    'directml' { 'directml' }
    default { 'cpu' }
}

if ($hostAddr -notin @('127.0.0.1', 'localhost')) {
    Write-Warning "FaceFusion binding to $hostAddr - ensure this is intentional."
}

$env:GRADIO_SERVER_NAME = $hostAddr
$env:GRADIO_SERVER_PORT = $port

$launchArgs = @(
    'run',
    '--config-path', $configPath,
    '--temp-path', $tempDir,
    '--output-path', $outputDir,
    '--execution-providers', $providers,
    '--execution-device-ids', '0'
)

$providerList = & $python -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))" 2>$null

Write-Host '=== FaceFusion Launch ==='
Write-Host "Version:   $($cfg.Lock.upstreams.facefusion.release)"
Write-Host "Env:       $envDir"
Write-Host "Provider:  $ExecutionProvider (available: $providerList)"
Write-Host "UI:        http://${hostAddr}:${port}"
Write-Host "Input:     $(Resolve-GenAIPath $root $cfg.Paths.facefusion.inputDir)"
Write-Host "Output:    $outputDir"
Write-Host "Temp:      $tempDir"
Write-Host "Logs:      $(Resolve-GenAIPath $root $cfg.Paths.facefusion.logDir)"
Write-Host '========================='

Push-Location $cloneDir
try {
    & $python (Join-Path $cloneDir 'facefusion.py') @launchArgs
} finally {
    Pop-Location
}

