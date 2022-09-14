#!/bin/bash
###########################################
# Change the variables below to your need:#
###########################################

# Install directory without trailing slash
install_dir="/home/$(whoami)"

# Name of the subdirectory (defaults to stable-diffusion-webui)
clone_dir="stable-diffusion-webui"

# Commandline arguments for webui.py, for example: export COMMANDLINE_ARGS=(--medvram --opt-split-attention)
export COMMANDLINE_ARGS=()

# python3 executable
python_cmd="python3"

# git executable
export GIT=""

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
venv_dir="venv"

# install command for torch
export TORCH_COMMAND=(python3 -m pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113)

# Requirements file to use for stable-diffusion-webui
export REQS_FILE=""

# Fixed git repos
export K_DIFFUSION_PACKAGE=""
export GFPGAN_PACKAGE=""

# Fixed git commits
export STABLE_DIFFUSION_COMMIT_HASH=""
export TAMING_TRANSFORMERS_COMMIT_HASH=""
export CODEFORMER_COMMIT_HASH=""
export BLIP_COMMIT_HASH=""

###########################################