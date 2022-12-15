$env:PYTHON= ''
$env:GIT= ''
$env:VENV_DIR= ''

# Commandline arguments for webui.py, for example: $env:COMMANDLINE_ARGS="--medvram --opt-split-attention"
$env:COMMANDLINE_ARGS=""

# script to launch to start the app
# $env:LAUNCH_SCRIPT="launch.py"

# install command for torch
# $env:TORCH_COMMAND="pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
# $env:REQS_FILE="requirements_versions.txt"

# $env:GFPGAN_PACKAGE=""
# $env:CLIP_PACKAGE=""
# $env:OPENCLIP_PACKAGE=""

# URL to a WHL if you wish to override default xformers windows
# $env:XFORMERS_WINDOWS_PACKAGE=""

# Uncomment and set to enable an alternate repository URL
# $env:STABLE_DIFFUSION_REPO=""
# $env:TAMING_TRANSFORMERS_REPO=""
# $env:K_DIFFUSION_REPO=""
# $env:CODEFORMER_REPO=""
# $env:BLIP_REPO=""

# Uncomment and set to enable a specific revision of a repository
# $env:STABLE_DIFFUSION_COMMIT_HASH=""
# $env:TAMING_TRANSFORMERS_COMMIT_HASH=""
# $env:K_DIFFUSION_COMMIT_HASH=""
# $env:CODEFORMER_COMMIT_HASH=""
# $env:BLIP_COMMIT_HASH=""


# Uncomment to enable accelerated launch
# $env:ACCELERATE="True"

$SCRIPT = "$PSScriptRoot\webui.ps1"
Invoke-Expression "$SCRIPT"
