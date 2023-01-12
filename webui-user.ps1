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
