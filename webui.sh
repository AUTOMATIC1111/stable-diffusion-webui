#! /bin/bash

PYTHON=python3
GIT=git
COMMANDLINE_ARGS=$@
VENV_DIR=venv
REQS_FILE=requirements.txt

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv $VENV_DIR
    source $VENV_DIR/bin/activate

    pip install --upgrade pip
    pip install wheel

    pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary
    pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary --only-binary=psutil

    pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary
    pip install -r $REQS_FILE --prefer-binary
    pip install -U numpy --prefer-binary
else
    source $VENV_DIR/bin/activate
fi

if [ ! -d "repositories" ]; then
    mkdir repositories

    $GIT clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
    $GIT clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
    $GIT clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer

    pip install -r repositories/CodeFormer/requirements.txt --prefer-binary
fi

if [ ! -f "model.ckpt" ] && [ ! -f "repositories/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt" ]; then
    echo Stable Diffusion model not found: you need to place model.ckpt file into same directory as this file.
    exit 1
fi

if [ ! -f "GFPGANv1.3.pth" ] && [ ! -f "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth" ]; then
    echo GFPGAN not found: you need to place GFPGANv1.3.pth file into same directory as this file.
    echo Face fixing feature will not work.
fi

echo Launching webui.py...
python webui.py $COMMANDLINE_ARGS
exit 0
