#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstrap FaceFusion into GenAI/runtime with an isolated Python venv.
#>
param(
    [ValidateSet('cuda', 'directml', 'cpu')]
    [string]$ExecutionProvider = ''
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$cfg = Get-GenAIConfig
Ensure-GenAIDirectories $cfg

$root = $cfg.Root
$log = Join-Path (Resolve-GenAIPath $root $cfg.Paths.facefusion.logDir) 'setup.log'
$cloneDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.cloneDir
$envDir = Resolve-GenAIPath $root $cfg.Paths.facefusion.envDir
$ff = $cfg.Lock.upstreams.facefusion
$configExample = Join-Path $root 'config\facefusion.example.ini'
$configLocal = Join-Path $root 'config\facefusion.ini'

if (-not $ExecutionProvider) {
    $ExecutionProvider = Get-SettingOrDefault $cfg.Settings 'FACEFUSION_EXECUTION_PROVIDER' 'cuda'
}

Write-GenAILog $log "Starting FaceFusion setup (provider: $ExecutionProvider)"

if (-not (Test-CommandExists 'git')) { throw 'Git is required.' }
if (-not (Test-CommandExists 'ffmpeg')) {
    Write-Warning 'FFmpeg not found in PATH. Install from https://ffmpeg.org/download.html or winget install ffmpeg'
}

$py = Get-PythonCandidate
if (-not $py) { throw 'Python 3.10+ is required.' }

Invoke-GitClonePinned -Repository $ff.repository -Commit $ff.commit -TargetDir $cloneDir -LogFile $log

if (-not (Test-Path (Join-Path $envDir 'Scripts\python.exe'))) {
    if ($py.Arg) { & $py.Command $py.Arg -m venv $envDir }
    else { & $py.Command -m venv $envDir }
}

$python = Join-Path $envDir 'Scripts\python.exe'
$pip = Join-Path $envDir 'Scripts\pip.exe'

Invoke-PipInstall -Pip $python -PipArgs @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools') -LogFile $log -Description 'pip bootstrap'
Invoke-PipInstall -Pip $pip -PipArgs @('install', '-r', (Join-Path $cloneDir 'requirements.txt')) -LogFile $log -Description 'FaceFusion requirements'

switch ($ExecutionProvider) {
    'cuda' {
        Invoke-PipInstall -Pip $pip -PipArgs @('install', 'onnxruntime-gpu') -LogFile $log -Description 'onnxruntime-gpu'
    }
    'directml' {
        Invoke-PipInstall -Pip $pip -PipArgs @('install', 'onnxruntime-directml') -LogFile $log -Description 'onnxruntime-directml'
    }
    default {
        Write-GenAILog $log 'Using CPU onnxruntime from requirements.txt'
    }
}

$tempPath = Resolve-GenAIPath $root $cfg.Paths.facefusion.tempDir
$outputPath = Resolve-GenAIPath $root $cfg.Paths.facefusion.outputDir
if (-not (Test-Path $configLocal)) {
    Copy-Item $configExample $configLocal
}
$content = Get-Content $configLocal -Raw
$content = $content -replace '(?m)^temp_path\s*=.*', "temp_path = $($tempPath -replace '\\', '/')"
$content = $content -replace '(?m)^output_path\s*=.*', "output_path = $($outputPath -replace '\\', '/')"
Set-Content -Path $configLocal -Value $content.TrimEnd() -Encoding UTF8
Write-GenAILog $log "Updated config/facefusion.ini paths"

$providerCheck = & $python -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
Write-GenAILog $log "ONNX providers: $providerCheck"

Write-Host "FaceFusion setup complete."
Write-Host "  Clone:    $cloneDir"
Write-Host "  Env:      $envDir"
Write-Host "  Provider: $ExecutionProvider ($providerCheck)"
Write-Host "  Log:      $log"
