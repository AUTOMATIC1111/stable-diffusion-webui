#!/bin/bash
####################################################################
#                          macOS defaults                          #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

export install_dir="$HOME"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --no-half-vae --use-cpu interrogate"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ "$(sysctl -n machdep.cpu.brand_string)" =~ ^.*"Intel".*$ ]]; then
    export TORCH_COMMAND="pip install torch==2.1.2 torchvision==0.16.2"
else
    export PIP_CONSTRAINT="${SCRIPT_DIR}/requirements_macos.txt"
    # Direct Metal matrix multiplication is faster than MPSGraph for the
    # projection sizes used by Stable Diffusion 1.x on M1.
    export PYTORCH_MPS_PREFER_METAL=1
    # PyTorch 2.3 is the newest runtime verified to render correctly on the
    # current macOS beta. The local MFA installer backports stream safety.
    export TORCH_COMMAND="pip install torch==2.3.1 torchvision==0.18.1"
    export MPS_FLASH_ATTENTION_INSTALLER="${SCRIPT_DIR}/scripts/install_mps_flash_attention.py"
fi

####################################################################
