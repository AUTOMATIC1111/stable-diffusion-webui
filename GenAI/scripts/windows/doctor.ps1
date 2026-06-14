#Requires -Version 5.1
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\..\lib\GenAI-Common.ps1"

$results = @()
$cfg = Get-GenAIConfig
$root = $cfg.Root

try {
    $results += Get-DiagnosticLine 'Repository path' 'PASS' $root
} catch {
    $results += Get-DiagnosticLine 'Repository path' 'FAIL' $_.Exception.Message
}

foreach ($f in @('config\local-paths.json', 'config\settings.env', 'upstreams.lock.json')) {
    $p = Join-Path $root $f
    if (Test-Path $p) { $results += Get-DiagnosticLine "Config: $f" 'PASS' }
    else { $results += Get-DiagnosticLine "Config: $f" 'FAIL' 'Run setup-all.ps1 to create from examples' }
}

if (Test-CommandExists 'git') { $results += Get-DiagnosticLine 'Git' 'PASS' (git --version) }
else { $results += Get-DiagnosticLine 'Git' 'FAIL' 'Install Git for Windows' }

$py = Get-PythonCandidate
if ($py) { $results += Get-DiagnosticLine 'Python' 'PASS' "$($py.Command) $($py.Version)" }
else { $results += Get-DiagnosticLine 'Python' 'FAIL' 'Install Python 3.10+' }

if (Test-CommandExists 'ffmpeg') { $results += Get-DiagnosticLine 'FFmpeg' 'PASS' (ffmpeg -version 2>&1 | Select-Object -First 1) }
else { $results += Get-DiagnosticLine 'FFmpeg' 'WARN' 'Required for FaceFusion video. winget install ffmpeg' }

# OS / GPU
$results += Get-DiagnosticLine 'OS' 'PASS' "$([System.Environment]::OSVersion.VersionString)"
if (Test-CommandExists 'nvidia-smi') {
    $gpu = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    $results += Get-DiagnosticLine 'NVIDIA GPU' 'PASS' ($gpu -join '; ')
} else {
    $results += Get-DiagnosticLine 'NVIDIA GPU' 'WARN' 'nvidia-smi not found'
}

# Disk
$drive = (Split-Path $root -Qualifier)
$free = (Get-PSDrive ($drive.TrimEnd(':'))).Free / 1GB
if ($free -gt 50) { $results += Get-DiagnosticLine 'Disk space' 'PASS' ("{0:N1} GB free" -f $free) }
elseif ($free -gt 20) { $results += Get-DiagnosticLine 'Disk space' 'WARN' ("{0:N1} GB free - models need more" -f $free) }
else { $results += Get-DiagnosticLine 'Disk space' 'FAIL' ("{0:N1} GB free" -f $free) }

# ComfyUI env
$comfyPy = Join-Path (Resolve-GenAIPath $root $cfg.Paths.comfyui.envDir) 'Scripts\python.exe'
$cloneComfy = Resolve-GenAIPath $root $cfg.Paths.comfyui.cloneDir
if (Test-Path $comfyPy) {
    $results += Get-DiagnosticLine 'ComfyUI venv' 'PASS' $comfyPy
    try {
        $tv = & $comfyPy -c "import torch; print(torch.__version__)"
        $cuda = & $comfyPy -c "import torch; print(torch.cuda.is_available())"
        $results += Get-DiagnosticLine 'ComfyUI PyTorch' 'PASS' "v$tv cuda=$cuda"
        if ($cuda -eq 'False') { $results += Get-DiagnosticLine 'ComfyUI CUDA' 'WARN' 'CPU-only PyTorch detected on Windows' }
        else { $results += Get-DiagnosticLine 'ComfyUI CUDA' 'PASS' (& $comfyPy -c "import torch; print(torch.cuda.get_device_name(0))") }
    } catch {
        $results += Get-DiagnosticLine 'ComfyUI PyTorch' 'FAIL' $_.Exception.Message
    }
} else {
    $results += Get-DiagnosticLine 'ComfyUI venv' 'FAIL' 'Run setup-comfyui.ps1'
}

if (Test-Path (Join-Path $cloneComfy 'main.py')) {
    $actual = Test-PinnedClone $cloneComfy $cfg.Lock.upstreams.comfyui.commit
    if ($actual -eq $cfg.Lock.upstreams.comfyui.commit) {
        $results += Get-DiagnosticLine 'ComfyUI pin' 'PASS' "$($cfg.Lock.upstreams.comfyui.release) @ $($actual.Substring(0,7))"
    } elseif ($actual) {
        $results += Get-DiagnosticLine 'ComfyUI pin' 'WARN' "Expected $($cfg.Lock.upstreams.comfyui.commit.Substring(0,7)) got $($actual.Substring(0,7))"
    } else {
        $results += Get-DiagnosticLine 'ComfyUI clone' 'PASS' $cfg.Lock.upstreams.comfyui.release
    }
} else {
    $results += Get-DiagnosticLine 'ComfyUI clone' 'FAIL' 'Missing runtime clone'
}

# FaceFusion env
$ffPy = Join-Path (Resolve-GenAIPath $root $cfg.Paths.facefusion.envDir) 'Scripts\python.exe'
$cloneFf = Resolve-GenAIPath $root $cfg.Paths.facefusion.cloneDir
if (Test-Path $ffPy) {
    $results += Get-DiagnosticLine 'FaceFusion venv' 'PASS' $ffPy
    try {
        $prov = & $ffPy -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
        $results += Get-DiagnosticLine 'FaceFusion ONNX providers' 'PASS' $prov
    } catch {
        $results += Get-DiagnosticLine 'FaceFusion ONNX' 'FAIL' $_.Exception.Message
    }
} else {
    $results += Get-DiagnosticLine 'FaceFusion venv' 'FAIL' 'Run setup-facefusion.ps1'
}

if (Test-Path (Join-Path $cloneFf 'facefusion.py')) {
    $actual = Test-PinnedClone $cloneFf $cfg.Lock.upstreams.facefusion.commit
    if ($actual -eq $cfg.Lock.upstreams.facefusion.commit) {
        $results += Get-DiagnosticLine 'FaceFusion pin' 'PASS' "$($cfg.Lock.upstreams.facefusion.release) @ $($actual.Substring(0,7))"
    } elseif ($actual) {
        $results += Get-DiagnosticLine 'FaceFusion pin' 'WARN' "Expected $($cfg.Lock.upstreams.facefusion.commit.Substring(0,7)) got $($actual.Substring(0,7))"
    } else {
        $results += Get-DiagnosticLine 'FaceFusion clone' 'PASS' $cfg.Lock.upstreams.facefusion.release
    }
} else {
    $results += Get-DiagnosticLine 'FaceFusion clone' 'FAIL' 'Missing runtime clone'
}

$ffPort = [int](Get-SettingOrDefault $cfg.Settings 'FACEFUSION_PORT' '7861')
if ($ffPort -eq 7860) {
    $results += Get-DiagnosticLine 'A1111 port conflict' 'WARN' 'FACEFUSION_PORT=7860 conflicts with parent WebUI default; use 7861'
}

# Ports
foreach ($entry in @(@('ComfyUI port', $cfg.Settings['COMFYUI_PORT']), @('FaceFusion port', $cfg.Settings['FACEFUSION_PORT']))) {
    $port = [int]$entry[1]
    try {
        $inUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($inUse) { $results += Get-DiagnosticLine $entry[0] 'WARN' "Port $port in use" }
        else { $results += Get-DiagnosticLine $entry[0] 'PASS' "Port $port available" }
    } catch {
        $results += Get-DiagnosticLine $entry[0] 'SKIP' "Port check unavailable: $($_.Exception.Message)"
    }
}

Write-DiagnosticReport $results

