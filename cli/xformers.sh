#/bin/env bash
echo "Installing xformers"

NVCC_FLAGS="--use_fast_math" 
FORCE_CUDA="1"
TORCH_CUDA_ARCH_LIST="8.6"
pip install ninja -q
pip uninstall xformers -y 2>/dev/null
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
pip show torch
pip show xformers
python -m xformers.info
