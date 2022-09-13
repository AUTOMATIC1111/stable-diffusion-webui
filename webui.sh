#!/bin/sh

# ----- Variables -----
# Commands
[ -n "$GIT" ]         || GIT="git"
[ -n "$PYTHON" ]      || PYTHON="python"
[ -n "$PIP" ]         || PIP="pip"
[ -n "$PIPINSTALL" ]  || PIPINSTALL="$PIP install --prefer-binary"

command -v "$GIT" > /dev/null    || exit_with_error "$GIT could not be found"
command -v "$PYTHON" > /dev/null || exit_with_error "$PYTHON could not be found"
command -v "$PIP" > /dev/null    || exit_with_error "$PYTHON could not be found"

# Git repositories
[ -n "$REPOSITORIES" ] || REPOSITORIES="$PWD/repositories"
[ -n "$SD_REPO" ]      || SD_REPO="https://github.com/CompVis/stable-diffusion.git"
[ -n "$SD_DIR" ]       || SD_DIR="$REPOSITORIES/stable-diffusion"
[ -n "$TT_REPO" ]      || TT_REPO="https://github.com/CompVis/taming-transformers.git"
[ -n "$TT_DIR" ]       || TT_DIR="$REPOSITORIES/taming-transformers"
[ -n "$CF_REPO" ]      || CF_REPO="https://github.com/sczhou/CodeFormer.git"
[ -n "$CF_DIR" ]       || CF_DIR="$REPOSITORIES/CodeFormer"
[ -n "$BL_REPO" ]      || BL_REPO="https://github.com/salesforce/BLIP.git"
[ -n "$BL_DIR" ]       || BL_DIR="$REPOSITORIES/BLIP"

# Files
[ -n "$REQS_FILE" ]    || REQS_FILE="$PWD/requirements.txt"
[ -n "$MODEL_FILE" ]   || MODEL_FILE="$PWD/model.ckpt"
[ -n "$GFPGAN_FILE" ]  || GFPGAN_FILE="$PWD/GFPGANv1.3.pth"

# ----- Functions -----
exit_with_error() {
    echo "$1"
    exit
}

# ----- Clone Necessary Repositories -----
[ ! -d "$REPOSITORIES" ] && mkdir -p "$REPOSITORIES"

if [ -d "$SD_DIR" ] ; then
    echo "Stable Diffusion is present"
else
    echo "Cloning Stable Difusion repository"
    "$GIT" clone "$SD_REPO" "$SD_DIR"
fi

if [ -d "$TT_DIR" ] ; then
    echo "Taming Transformers is present"
else
    echo "Cloning Taming Transformers repository"
    "$GIT" clone "$TT_REPO" "$TT_DIR"
fi

if [ -d "$CF_DIR" ] ; then
    echo "CodeFormer is present"
else
    echo "Cloning CodeFormer repository"
    "$GIT" clone "$CF_REPO" "$CF_DIR"
fi

if [ -d "$BL_DIR" ] ; then
    echo "BLIP is present"
else
    echo "Cloning BLIP repository"
    "$GIT" clone "$BL_REPO" "$BL_DIR"
    "$GIT" -C repositories/BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9
fi

# ----- Install Dependencies -----
if "$PYTHON" -c "import torch" ; then
    echo "pytorch and pyvision are present"
else
    NUMPY=1
    echo "Installing pytorch and pyvision"
    $PIPINSTALL torch torchvision
fi

if "$PYTHON" -c "import transformers; import wheel" ; then
    echo "Stanble Diffusion requirements are present"
else
    NUMPY=1
    echo "Installing Stanble Diffusion requirements"
    $PIPINSTALL transformers==4.19.2 diffusers invisible-watermark
fi

if "$PYTHON" -c "import k_diffusion.sampling" ; then
    echo "K-Diffusion is present"
else
    NUMPY=1
    echo "Installing K-Diffusion"
    $PIPINSTALL git+https://github.com/crowsonkb/k-diffusion.git
fi

if "$PYTHON" -c "import gfpgan" ; then
    echo "GFPGAN is present"
else
    NUMPY=1
    echo "Installing GFPGAN"
    $PIPINSTALL git+https://github.com/TencentARC/GFPGAN.git
fi

if "$PYTHON" -c "import lpips" ; then
    echo "CodeFormer requirements are present"
else
    NUMPY=1
    echo "Installing CodeFormer requirements"
    $PIPINSTALL -r "$CF_DIR/requirements.txt"
fi

if "$PYTHON" -c "import omegaconf; import fonts; import timm" ; then
    echo "Webui requirements are present"
else
    NUMPY=1
    echo "Installing webui requirements"
    $PIPINSTALL -r "$REQS_FILE"
fi

if [ -n "$NUMPY" ] ; then
    echo "Updating numpy"
    $PIPINSTALL -U numpy
fi

# ----- Check models -----
if [ -s "$MODEL_FILE" ] ; then
    echo "Stable Diffusion model found."
else
    exit_with_error "Stable Diffusion model not found: you need to place $MODEL_FILE in $PWD."
fi

if [ -s "$GFPGAN_FILE" ] ; then
    echo "GFPGAN model found."
else
    exit_with_error "GFPGAN model not found: you need to place $GFPGAN_FILE in $PWD."
fi

# ----- Check Cuda -----
if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" ; then
    echo "CUDA is available"
else
    exit_with_error "CUDA is not available"
fi

# ----- Launch -----
echo "Cloned repositories:"
"$GIT" show-ref

echo "Launching webui.py"
"$PYTHON" webui.py "$@"
