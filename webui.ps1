if ($env:PYTHON -eq "" -or $env:PYTHON -eq $null) {
    $PYTHON = "Python.exe"
} else {
    $PYTHON = $env:PYTHON
}

if ($env:VENV_DIR -eq "" -or $env:VENV_DIR -eq $null) {
    $VENV_DIR = "$PSScriptRoot\venv"
} else {
    $VENV_DIR = $env:VENV_DIR
}

if ($env:LAUNCH_SCRIPT -eq "" -or $env:LAUNCH_SCRIPT -eq $null) {
    $LAUNCH_SCRIPT = "$PSScriptRoot\launch.py"
} else {
    $LAUNCH_SCRIPT = $env:LAUNCH_SCRIPT
}

$ERROR_REPORTING = $false

mkdir tmp 2>$null

function Start-Venv {
    if ($VENV_DIR -eq '-') {
        Skip-Venv
    }

    if (Test-Path -Path "$VENV_DIR\Scripts\$python") {
        Activate-Venv
    } else {
        $PYTHON_FULLNAME = & $PYTHON -c "import sys; print(sys.executable)"
        Write-Output "Creating venv in directory $VENV_DIR using python $PYTHON_FULLNAME"
        Invoke-Expression "$PYTHON_FULLNAME -m venv $VENV_DIR > tmp/stdout.txt 2> tmp/stderr.txt"
        if ($LASTEXITCODE -eq 0) {
            Activate-Venv
        } else {
            Write-Output "Unable to create venv in directory $VENV_DIR"
        }
    }
}

function Activate-Venv {
    $PYTHON = "$VENV_DIR\Scripts\Python.exe"
    $ACTIVATE = "$VENV_DIR\Scripts\activate.bat"
    Invoke-Expression "cmd.exe /c $ACTIVATE"
    Write-Output "Venv set to $VENV_DIR."
    if ($ACCELERATE -eq 'True') {
        Check-Accelerate
    } else {
        Launch-App
    }
}

function Skip-Venv {
    Write-Output "Venv set to $VENV_DIR."
    if ($ACCELERATE -eq 'True') {
        Check-Accelerate
    } else {
        Launch-App
    }
}

function Check-Accelerate {
    Write-Output 'Checking for accelerate'
    $ACCELERATE = "$VENV_DIR\Scripts\accelerate.exe"
    if (Test-Path -Path $ACCELERATE) {
        Accelerate-Launch
    } else {
        Launch-App
    }
}

function Launch-App {
    Write-Output "Launching with python"
    Invoke-Expression "$PYTHON $LAUNCH_SCRIPT"
    #pause
    exit
}

function Accelerate-Launch {
    Write-Output 'Accelerating'
    Invoke-Expression "$ACCELERATE launch --num_cpu_threads_per_process=6 $LAUNCH_SCRIPT"
    #pause
    exit
}


try {
    if(Get-Command $PYTHON){
        Start-Venv
    }
} Catch {
    Write-Output "Couldn't launch python."
}