# Automatic Installation
## Windows (method 1)
> A very basic guide to get Stable Diffusion web UI up and running on Windows 10/11 NVIDIA GPU.
1. Download the `sd.webui.zip` from [v1.2.1](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.2.1) and extract the zip file.
2. Double click the `update.bat` script to update web UI to the latest version, wait till finish then close the window.
3. Double click the `run.bat` script to launch web UI, during the first launch it will download large amounts of files. After everything has been downloaded and installed correctly, you should see a message "`Running on local URL:  http://127.0.0.1:7860`", opening the link will present you with the web UI interface.
> you should be able to start generating images

### Extra configurations via `COMMANDLINE_ARGS`
There are some configuration options that you may want apply to web UI, in order to configure these options you need to edit the launch script found at `sd.webui\webui\webui-user.bat`, edit the file add the selected arguments after `set COMMANDLINE_ARGS=` like so :
```bat
set COMMANDLINE_ARGS=--autolaunch --update-check
```
> Each individual argument need to separated by a space, the above example well configure web UI to auto launch the browser page after it completes loading, and also chack for new version of web UI at launch.

### Troubleshooting
The default configuration of web UI should run on most modon GPU, but in some cases you may need some extra arguments to allow it to work properly.

1. For GPU with less ammounts of VRAM, you may need `--medvram` or `--lowvram`, these optimizations reduces VRAM requirements but sacrifice performance. If you do not have enough VRAM, web UI may refuse to launch or fail to generate images due to an out-of-memory error. The amount of VRAM required largely depends the desired image resolution, for more details see [Troubleshooting](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Troubleshooting).
> the [Tiled VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111) extension can help to reduce the VRAM requirement.

2. If your generated results in resulting in a black or green image, try adding `--precision full` and `--no-half`. 

3. Some combinations of model and VAE are prone to produce `NansException: A tensor with all NaNs was produced in VAE` resulting in a black image, using the option `--no-half-vae` may help to mitigate this issue.

### Extra Options
1. There are several cross attenuation optimization methods such as `--xformers` or `--opt-sdp-attention`, these can drastically increase performance see [Optimizations](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations) for more details, experiment with different options as different hardware are suited for different optimizations. If you wish to measure your system's performance, try using [sd-extension-system-info](https://github.com/vladmandic/sd-extension-system-info) extension which features a benchmarking tool and a [database](https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html) of user submitted results.
2. add `--autolaunch` to heve web UI launch the web browser altomaticly after web UI has started up.
3. add `--update-check` will notify you when theres a new version of webui.
4. See [Command Line Arguments and Settings
](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings) for more configuration options.

### Tip
If you already have stable diffusion models downloaded, you can move the models into `sd.webui\webui\models\Stable-diffusion\` before running `run.bat` in step 3, this will skip auto downloading the vanilla [stable-diffusion-v1-5 model](https://huggingface.co/runwayml/stable-diffusion-v1-5) model.

## Windows (method 2)
1. Install [Python 3.10.6](https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe) (ticking **Add to PATH**), and [git](https://github.com/git-for-windows/git/releases/download/v2.39.2.windows.1/Git-2.39.2-64-bit.exe)
2. Open Command Prompt from search bar, and type `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
3. Double click `webui-user.bat`

Installation video in case you get stuck: \
<sup>solves [#8229](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8229)</sup>

<details><summary>Video: (Click to expand:)</summary>

https://user-images.githubusercontent.com/98228077/223032534-c5dd5b13-a4b6-47a7-995c-27ed8ba8b3e7.mp4

</details>

<details><summary>Alternative Powershell launch scripts:</summary>

**webui.ps1**

```
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
```


**webui-user.ps1**

```
[Environment]::SetEnvironmentVariable("PYTHON", "")
[Environment]::SetEnvironmentVariable("GIT", "")
[Environment]::SetEnvironmentVariable("VENV_DIR","")

# Commandline arguments for webui.py, for example: [Environment]::SetEnvironmentVariable("COMMANDLINE_ARGS", "--medvram --opt-split-attention")
[Environment]::SetEnvironmentVariable("COMMANDLINE_ARGS", "")

# script to launch to start the app
# [Environment]::SetEnvironmentVariable("LAUNCH_SCRIPT", "launch.py")

# install command for torch
# [Environment]::SetEnvironmentVariable("TORCH_COMMAND", "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")

# Requirements file to use for stable-diffusion-webui
# [Environment]::SetEnvironmentVariable("REQS_FILE", "requirements_versions.txt")

# [Environment]::SetEnvironmentVariable("GFPGAN_PACKAGE", "")
# [Environment]::SetEnvironmentVariable("CLIP_PACKAGE", "")
# [Environment]::SetEnvironmentVariable("OPENCLIP_PACKAGE", "")

# URL to a WHL if you wish to override default xformers windows
# [Environment]::SetEnvironmentVariable("XFORMERS_WINDOWS_PACKAGE", "")

# Uncomment and set to enable an alternate repository URL
# [Environment]::SetEnvironmentVariable("STABLE_DIFFUSION_REPO", "")
# [Environment]::SetEnvironmentVariable("TAMING_TRANSFORMERS_REPO", "")
# [Environment]::SetEnvironmentVariable("K_DIFFUSION_REPO", "")
# [Environment]::SetEnvironmentVariable("CODEFORMER_REPO", "")
# [Environment]::SetEnvironmentVariable("BLIP_REPO", "")

# Uncomment and set to enable a specific revision of a repository
# [Environment]::SetEnvironmentVariable("STABLE_DIFFUSION_COMMIT_HASH", "")
# [Environment]::SetEnvironmentVariable("TAMING_TRANSFORMERS_COMMIT_HASH", "")
# [Environment]::SetEnvironmentVariable("K_DIFFUSION_COMMIT_HASH", "")
# [Environment]::SetEnvironmentVariable("CODEFORMER_COMMIT_HASH", "")
# [Environment]::SetEnvironmentVariable("BLIP_COMMIT_HASH", "")


# Uncomment to enable accelerated launch
# [Environment]::SetEnvironmentVariable("ACCELERATE", "True")

$SCRIPT = "$PSScriptRoot\webui.ps1"
Invoke-Expression "$SCRIPT"
```

</details>





See [Troubleshooting](Troubleshooting) section for what to do if things go wrong.



## Third party installation guides/scripts:
- NixOS: https://github.com/virchau13/automatic1111-webui-nix

# Almost Automatic Installation and Launch
To install the required packages via pip without creating a virtual environment, run:
```bash
python launch.py
```

Command line arguments may be passed directly, for example:
```bash
python launch.py --opt-split-attention --ckpt ../secret/anime9999.ckpt
```

# Manual Installation
Manual installation is very outdated and probably won't work. check colab in the repo's readme for instructions.

The following process installs everything manually on both Windows or Linux (the latter requiring `dir` to be replaced by `ls`):
```bash
# install torch with CUDA support. See https://pytorch.org/get-started/locally/ for more instructions if this fails.
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

# check if torch supports GPU; this must output "True". You need CUDA 11. installed for this. You might be able to use
# a different version, but this is what I tested.
python -c "import torch; print(torch.cuda.is_available())"

# clone web ui and go into its directory
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# clone repositories for Stable Diffusion and (optionally) CodeFormer
mkdir repositories
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# install requirements of Stable Diffusion
pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary

# install k-diffusion
pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary

# (optional) install GFPGAN (face restoration)
pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary

# (optional) install requirements for CodeFormer (face restoration)
pip install -r repositories/CodeFormer/requirements.txt --prefer-binary

# install requirements of web ui
pip install -r requirements.txt  --prefer-binary

# update numpy to latest version
pip install -U numpy  --prefer-binary

# (outside of command line) put stable diffusion model into web ui directory
# the command below must output something like: 1 File(s) 4,265,380,512 bytes
dir model.ckpt

```

The installation is finished, to start the web ui, run:
```bash
python webui.py
```

# Windows 11 WSL2 instructions
To install under a Linux distro in Windows 11's WSL2:
```bash
# install conda (if not already done)
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh

# Clone webui repo
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Create and activate conda env
conda env create -f environment-wsl2.yaml
conda activate automatic

```

At this point, the instructions for the Manual installation may be applied starting at step `# clone repositories for Stable Diffusion and (optionally) CodeFormer`.


# Alternative installation on Windows using Conda
- Prerequisites _*(Only needed if you do not have them)*_. Assumes [Chocolatey](https://chocolatey.org/install) is installed. 
    ```bash
    # install git
    choco install git
    # install conda
    choco install anaconda3
    ```
    Optional parameters: [git](https://community.chocolatey.org/packages/git), [conda](https://community.chocolatey.org/packages/anaconda3)
- Install (warning: some files exceed multiple gigabytes, make sure you have space first)
  1. Download as .zip and extract or use git to clone.
        ```bash
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
        ```
  2. Launch the Anaconda prompt. It should be noted that you can use older Python versions, but you may be forced to manually remove features like cache optimization, which will degrade your performance.
        ```bash
        # Navigate to the git directory
        cd "GIT\StableDiffusion"
        # Create environment
        conda create -n StableDiffusion python=3.10.6
        # Activate environment
        conda activate StableDiffusion
        # Validate environment is selected
        conda env list
        # Start local webserver
        webui-user.bat
        # Wait for "Running on local URL:  http://127.0.0.1:7860" and open that URI.
        ```
    3. _*(Optional)*_ Go to [CompVis](https://huggingface.co/CompVis) and download latest model, for example [1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and unpack it to ex:
        ```bash
        GIT\StableDiffusion\models\Stable-diffusion
        ```
        after that restart the server by restarting Anaconda prompt and 
        ```bash
        webui-user.bat
        ```
- Alternative defaults worth trying out:
    1. Try **euler a** (Ancestral Euler) with higher **Sampling Steps** ex: 40 or others with 100. 
    2. Set "Settings > User interface > Show image creation progress every N sampling steps" to 1 and pick a deterministic **Seed** value. Can visually see how image defusion happens and record a .gif with [ScreenToGif](https://github.com/NickeManarin/ScreenToGif).
    3. Use **Restore faces**. Generally, better results, but that quality comes at the cost of speed.

