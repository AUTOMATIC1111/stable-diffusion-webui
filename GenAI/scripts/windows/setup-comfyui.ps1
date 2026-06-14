#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstrap ComfyUI into GenAI/runtime with an isolated Python venv.
#>
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
$log = Join-Path (Resolve-GenAIPath $root $cfg.Paths.comfyui.logDir) 'setup.log'
$cloneDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.cloneDir
$envDir = Resolve-GenAIPath $root $cfg.Paths.comfyui.envDir
$comfy = $cfg.Lock.upstreams.comfyui
$torchIndex = $cfg.Lock.pytorch.windows_cuda_index

Write-GenAILog $log "Starting ComfyUI setup (profile: $Profile)"

if (-not (Test-CommandExists 'git')) {
    throw 'Git is required. Install from https://git-scm.com/download/win'
}

$py = Get-PythonCandidate
if (-not $py) { throw 'Python 3.10+ is required. Install from https://www.python.org/downloads/' }
Write-GenAILog $log "Using Python $($py.Version) via $($py.Command)"

Invoke-GitClonePinned -Repository $comfy.repository -Commit $comfy.commit -TargetDir $cloneDir -LogFile $log

if (-not (Test-Path (Join-Path $envDir 'Scripts\python.exe'))) {
    if ($py.Arg) { & $py.Command $py.Arg -m venv $envDir }
    else { & $py.Command -m venv $envDir }
    Write-GenAILog $log "Created venv at $envDir"
}

$python = Join-Path $envDir 'Scripts\python.exe'
$pip = Join-Path $envDir 'Scripts\pip.exe'

Invoke-PipInstall -Pip $python -PipArgs @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools') -LogFile $log -Description 'pip bootstrap'

Write-GenAILog $log "Installing PyTorch with CUDA from $torchIndex"
Invoke-PipInstall -Pip $pip -PipArgs @('install', 'torch', 'torchvision', 'torchaudio', '--index-url', $torchIndex) -LogFile $log -Description 'torch cu124'

$req = Join-Path $cloneDir 'requirements.txt'
Invoke-PipInstall -Pip $pip -PipArgs @('install', '-r', $req) -LogFile $log -Description 'ComfyUI requirements'

$cudaCheck = & $python -c "import torch; print('cuda=' + str(torch.cuda.is_available()) + ';device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"
Write-GenAILog $log "PyTorch check: $cudaCheck"
if ($cudaCheck -notmatch 'cuda=True') {
    Write-Warning 'CUDA not available. ComfyUI will run on CPU. Verify NVIDIA driver and cu124 wheel compatibility.'
}

$modelsRoot = Resolve-GenAIPath $root $cfg.Paths.comfyui.modelsDir
$externalRoot = $cfg.Settings['EXTERNAL_MODEL_ROOT']
$extraPaths = @"
comfyui:
  base_path: $($modelsRoot -replace '\\', '/')
  checkpoints: checkpoints
  vae: vae
  loras: loras
  embeddings: embeddings
  controlnet: controlnet
  upscale_models: upscale_models
"@
if ($externalRoot) {
    $extraPaths += "`nexternal:`n  base_path: $($externalRoot -replace '\\', '/')`n  checkpoints: checkpoints"
}
$extraPathsFile = Join-Path $cloneDir 'extra_model_paths.yaml'
Set-Content -Path $extraPathsFile -Value $extraPaths -Encoding UTF8
Write-GenAILog $log "Wrote extra_model_paths.yaml"

Write-Host "ComfyUI setup complete."
Write-Host "  Clone:  $cloneDir"
Write-Host "  Env:    $envDir"
Write-Host "  Models: $modelsRoot"
Write-Host "  Log:    $log"
