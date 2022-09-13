#!/bin/bash
###########################################
# Change the variables below to your need:#
###########################################

# Install directory without trailing slash
install_dir="/home/$(whoami)"

# Name of the subdirectory (defaults to stable-diffusion-webui)
clone_dir="stable-diffusion-webui"

# Commandline arguments for webui.py, for example: commandline_args=(--medvram --opt-split-attention)
commandline_args=()

# python3 executable
python_cmd="python3"

# pip3 executable
pip_cmd=(python3 -m pip)

# git executable
git_cmd="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
venv_dir="venv"

# pip3 install command for torch
torch_command=(torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113)

# Requirements file to use for stable-diffusion-webui
reqs_file="requirements_versions.txt"

###########################################