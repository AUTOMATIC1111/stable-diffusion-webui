# GenAI shared PowerShell helpers
# Dot-source from scripts in scripts/windows/

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-GenAIRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir '..\..')).Path
}

function Get-GenAIConfig {
    param([string]$Root = (Get-GenAIRoot))
    $pathsExample = Join-Path $Root 'config\local-paths.example.json'
    $pathsFile = Join-Path $Root 'config\local-paths.json'
    $settingsExample = Join-Path $Root 'config\settings.example.env'
    $settingsFile = Join-Path $Root 'config\settings.env'
    $lockFile = Join-Path $Root 'upstreams.lock.json'

    if (-not (Test-Path $pathsFile)) {
        Copy-Item $pathsExample $pathsFile
        Write-Host "Created config\local-paths.json from example."
    }
    if (-not (Test-Path $settingsFile)) {
        Copy-Item $settingsExample $settingsFile
        Write-Host "Created config\settings.env from example."
    }

    $paths = Get-Content $pathsFile -Raw | ConvertFrom-Json
    $settings = @{}
    Get-Content $settingsFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
        $parts = $_ -split '=', 2
        if ($parts.Count -eq 2) { $settings[$parts[0].Trim()] = $parts[1].Trim() }
    }
    $lock = Get-Content $lockFile -Raw | ConvertFrom-Json

    return [PSCustomObject]@{
        Root = $Root
        Paths = $paths
        Settings = $settings
        Lock = $lock
    }
}

function Resolve-GenAIPath {
    param([string]$Root, [string]$RelativePath)
    return (Join-Path $Root ($RelativePath -replace '/', '\'))
}

function Ensure-GenAIDirectories {
    param($Config)
    $dirs = @(
        $Config.Paths.comfyui.cloneDir,
        $Config.Paths.comfyui.envDir,
        $Config.Paths.comfyui.modelsDir,
        $Config.Paths.comfyui.inputDir,
        $Config.Paths.comfyui.outputDir,
        $Config.Paths.comfyui.logDir,
        $Config.Paths.facefusion.cloneDir,
        $Config.Paths.facefusion.envDir,
        $Config.Paths.facefusion.modelsDir,
        $Config.Paths.facefusion.inputDir,
        $Config.Paths.facefusion.outputDir,
        $Config.Paths.facefusion.tempDir,
        $Config.Paths.facefusion.logDir,
        'models/comfyui/checkpoints',
        'models/comfyui/vae',
        'models/comfyui/loras',
        'models/comfyui/embeddings',
        'models/comfyui/controlnet',
        'models/comfyui/upscale_models',
        'models/facefusion',
        'workflows/comfyui',
        'workflows/facefusion'
    )
    foreach ($d in $dirs) {
        $full = Resolve-GenAIPath $Config.Root $d
        if (-not (Test-Path $full)) {
            New-Item -ItemType Directory -Path $full -Force | Out-Null
        }
    }
}

function Write-GenAILog {
    param([string]$Path, [string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -Path $Path -Value $line
    Write-Host $line
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-PipInstall {
    param(
        [string]$Pip,
        [string[]]$PipArgs,
        [string]$LogFile,
        [string]$Description
    )
    Write-GenAILog $LogFile "pip $Description"
    $output = & $Pip @PipArgs 2>&1
    $output | ForEach-Object { Add-Content -Path $LogFile -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed for $Description (exit code $LASTEXITCODE). See $LogFile"
    }
}

function Get-SettingOrDefault {
    param([hashtable]$Settings, [string]$Key, [string]$Default)
    if ($Settings.ContainsKey($Key) -and $Settings[$Key]) { return $Settings[$Key] }
    return $Default
}

function Test-PinnedClone {
    param([string]$CloneDir, [string]$ExpectedCommit)
    if (-not (Test-Path (Join-Path $CloneDir '.git'))) { return $null }
    Push-Location $CloneDir
    try {
        return (git rev-parse HEAD).Trim()
    } finally {
        Pop-Location
    }
}

function Get-PythonCandidate {
    foreach ($cmd in @('py -3.12', 'py -3.11', 'py -3.10', 'python3.12', 'python3.11', 'python3.10', 'python')) {
        $name, $arg = $cmd -split ' ', 2
        if (-not (Test-CommandExists $name)) { continue }
        try {
            if ($arg) {
                $v = & $name $arg -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            } else {
                $v = & $name -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            }
            if ($v) { return @{ Command = $name; Arg = $arg; Version = $v.Trim() } }
        } catch { }
    }
    return $null
}

function Invoke-GitClonePinned {
    param(
        [string]$Repository,
        [string]$Commit,
        [string]$TargetDir,
        [string]$LogFile
    )
    $prevErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
    if (Test-Path (Join-Path $TargetDir '.git')) {
        Push-Location $TargetDir
        try {
            git fetch --tags --quiet origin
            git checkout $Commit 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "git checkout $Commit failed" }
            Write-GenAILog $LogFile "Checked out pinned commit $Commit in $TargetDir"
        } finally {
            Pop-Location
        }
    } else {
        if (Test-Path $TargetDir) { Remove-Item $TargetDir -Recurse -Force }
        New-Item -ItemType Directory -Path (Split-Path $TargetDir) -Force | Out-Null
        git clone $Repository $TargetDir 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "git clone failed for $Repository" }
        Push-Location $TargetDir
        try {
            git checkout $Commit 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "git checkout $Commit failed" }
            Write-GenAILog $LogFile "Cloned and checked out $Commit"
        } finally {
            Pop-Location
        }
    }
    } finally {
        $ErrorActionPreference = $prevErrorAction
    }
}

function Get-DiagnosticLine {
    param([string]$Name, [string]$Status, [string]$Detail = '')
    return [PSCustomObject]@{ Check = $Name; Status = $Status; Detail = $Detail }
}

function Write-DiagnosticReport {
    param([array]$Results)
    $width = 52
    Write-Host ('=' * $width)
    Write-Host 'GenAI Doctor Report'
    Write-Host ('=' * $width)
    foreach ($r in $Results) {
        $color = switch ($r.Status) {
            'PASS' { 'Green' }
            'WARN' { 'Yellow' }
            'FAIL' { 'Red' }
            default { 'Gray' }
        }
        Write-Host ("[{0}] {1}" -f $r.Status, $r.Check) -ForegroundColor $color
        if ($r.Detail) { Write-Host "      $($r.Detail)" }
    }
    Write-Host ('=' * $width)
    if ($Results | Where-Object { $_.Status -eq 'FAIL' }) { exit 1 }
    if ($Results | Where-Object { $_.Status -eq 'WARN' }) { exit 2 }
    exit 0
}
