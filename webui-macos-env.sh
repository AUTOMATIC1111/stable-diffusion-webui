#!/bin/bash
####################################################################
#                          macOS defaults                          #
# Please modify webui-user.sh to change these instead of this file #
####################################################################

if [[ -x "$(command -v python3.10)" ]]
then
    python_cmd="python3.10"
fi

export install_dir="$HOME"
export COMMANDLINE_ARGS="--skip-torch-cuda-test --no-half --use-cpu interrogate"
export TORCH_COMMAND="pip install torch==1.12.1 torchvision==0.13.1"
export K_DIFFUSION_REPO="https://github.com/brkirch/k-diffusion.git"
export K_DIFFUSION_COMMIT_HASH="51c9778f269cedb55a4d88c79c0246d35bdadb71"
export PYTORCH_ENABLE_MPS_FALLBACK=1

####################################################################
