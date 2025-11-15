#!/bin/bash
##########################################################################################
# ROCm 6.2 Optimized Launch Script for AMD GPUs with 16GB VRAM
# Based on best practices for VRAM optimization and memory management
##########################################################################################

# Install directory without trailing slash
#install_dir="/home/$(whoami)"

# Name of the subdirectory
#clone_dir="stable-diffusion-webui"

# ============================================================================
# ROCm 6.2 PyTorch Installation
# ============================================================================
# Install PyTorch with ROCm 6.2 support
export TORCH_COMMAND="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"

# ============================================================================
# PyTorch HIP Memory Allocation Configuration
# ============================================================================
# Prevents memory fragmentation - CRITICAL for stable VRAM usage
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Command Line Arguments for VRAM Optimization (16GB VRAM)
# ============================================================================
# Explanation of flags:
#   --skip-torch-cuda-test   : Skip CUDA test (we're using ROCm/HIP)
#   --medvram                : Optimized for 8-16GB VRAM, moves models between GPU/CPU as needed
#   --opt-split-attention    : Reduces VRAM usage during attention computation
#   --no-half-vae            : Prevents VAE errors by using full precision for VAE
#
# Additional optional flags for extreme VRAM savings (uncomment if needed):
#   --lowvram                : For GPUs with <8GB VRAM (use instead of --medvram)
#   --xformers               : Use xformers for additional memory optimization (requires installation)
#   --opt-sdp-attention      : Alternative attention optimization
export COMMANDLINE_ARGS="--skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae"

# ============================================================================
# Optional: Additional VRAM Optimization Flags
# ============================================================================
# Uncomment the line below for more aggressive VRAM savings:
# export COMMANDLINE_ARGS="--skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae --opt-channelslast"

# Uncomment for extreme low VRAM mode (<8GB):
# export COMMANDLINE_ARGS="--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae"

# ============================================================================
# Python and Git Configuration
# ============================================================================
# python3 executable
#python_cmd="python3"

# git executable
#export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
#venv_dir="venv"

# script to launch to start the app
#export LAUNCH_SCRIPT="launch.py"

# ============================================================================
# Package Configuration
# ============================================================================
# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# ============================================================================
# Performance Tuning
# ============================================================================
# Uncomment to enable accelerated launch
#export ACCELERATE="True"

# Uncomment to disable TCMalloc (Thread-Caching Malloc)
# TCMalloc improves CPU memory allocation performance
#export NO_TCMALLOC="True"

##########################################################################################
# Usage Instructions:
#
# 1. Copy this file to webui-user.sh:
#    cp webui-user-rocm62.sh webui-user.sh
#
# 2. Launch the WebUI:
#    ./webui.sh
#
# 3. In WebUI Settings → Optimizations, configure:
#    - Enable quantization in K samplers: ✓
#    - Token merging ratio: 0.5
#    - Cross attention optimization: Doggettx (should be active)
#
# 4. Recommended Generation Settings for 16GB VRAM:
#
#    Safe Mode (no errors):
#      - Size: 512x512
#      - Hires fix: OFF
#      - Batch size: 1
#
#    Quality Mode (with upscaling):
#      - Size: 512x512
#      - Hires fix: ON
#        - Upscale by: 1.5 (not 2.0)
#        - Hires steps: 10
#        - Denoising: 0.4
#
#    With ControlNet:
#      - Size: 512x512
#      - Hires fix: OFF
#      - ControlNet units: max 1-2 active
#      - Low VRAM mode: ON in ControlNet settings
#
# 5. Workflow for Best Quality:
#    Phase 1 - Generation: 512x512, no hires fix → find perfect seed/prompt
#    Phase 2 - Upscaling: Use img2img or "Send to Extras" → R-ESRGAN 4x+
#
##########################################################################################
