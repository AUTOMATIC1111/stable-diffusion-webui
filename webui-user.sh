# #!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

#clean cache
rm -rf ~/.cache

# pull changes
git pull --recurse-submodules

# install metrics tool
apt update
apt install nano screen unzip -y
pip install aioprometheus[starlette]


# download models from gcp if needed, it can take while on a fresh disk.
echo "Checking model downloads..."
./rclone-v1.62.2-linux-amd64/rclone sync gs://ag-diffusion/models/Stable-diffusion models/Stable-diffusion --progress --config rclone-v1.62.2-linux-amd64/rclone.conf 
./rclone-v1.62.2-linux-amd64/rclone sync gs://ag-diffusion/models/lora models/Lora --progress --config rclone-v1.62.2-linux-amd64/rclone.conf 
./rclone-v1.62.2-linux-amd64/rclone copy gs://ag-diffusion/extensions/sd-webui-controlnet/models extensions/sd-webui-controlnet/models --progress --config rclone-v1.62.2-linux-amd64/rclone.conf 
./rclone-v1.62.2-linux-amd64/rclone sync gs://ag-diffusion/embeddings embeddings --progress --config rclone-v1.62.2-linux-amd64/rclone.conf 
echo "Checking model downloads: complete"

# Install directory without trailing slash
install_dir="/workspace"

# Name of the subdirectory
#clone_dir="stable-diffusion-webui"

#load env variable from .bashrc
source ~/.bashrc

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS="--medvram --opt-split-attention"
export COMMANDLINE_ARGS="--port 3000 --xformers --api --api-auth $AUTOMATIC1111_AUTH --api-log --nowebui --ckpt /workspace/stable-diffusion-webui/models/Stable-diffusion/deliberate_v2.safetensors --listen --enable-insecure-extension-access"
echo $COMMANDLINE_ARGS

# python3 executable
#python_cmd="python3"

# git executable
#export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
# venv_dir="/workspace/venv"

# script to launch to start the app
#export LAUNCH_SCRIPT="launch.py"

# install command for torch
#export TORCH_COMMAND="pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113"

# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export TAMING_TRANSFORMERS_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
#export ACCELERATE="True"

###########################################

