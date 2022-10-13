#!/bin/bash

# This script will install git and python (if not found on the PATH variable)
#  using micromamba (an 8mb static-linked single-file binary, conda replacement).
# For users who already have git and python, this step will be skipped.

# Next, it'll checkout the project's git repo, if necessary.
# Finally, it'll start webui-user.sh (to proceed with the usual installation).

# This enables a user to install this project without manually installing python and git.

OS_NAME=$(uname -s)
case "${OS_NAME}" in
    Linux*)     OS_NAME="linux";;
    Darwin*)    OS_NAME="mac";;
    *)          echo "Unknown OS: $OS_NAME! This script runs only on Linux or Mac" && exit
esac

OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*)    OS_ARCH="x64";;
    arm64*)     OS_ARCH="arm64";;
    *)          echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64" && exit
esac

# config
export MAMBA_ROOT_PREFIX="$(pwd)/installer_files/mamba"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"
MICROMAMBA_BINARY_FILE="$(pwd)/installer_files/micromamba_${OS_NAME}_${OS_ARCH}"
if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

# figure out what needs to be installed
PACKAGES_TO_INSTALL=""

if ! hash "python" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL python"; fi
if ! hash "git" &>/dev/null; then PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL git"; fi

# install git and python into a contained environment (if necessary)
if [ "$PACKAGES_TO_INSTALL" != "" ]; then
    echo "Packages to install: $PACKAGES_TO_INSTALL"

    # initialize the mamba binary
    mkdir -p "$MAMBA_ROOT_PREFIX"
    cp "$MICROMAMBA_BINARY_FILE" "$MAMBA_ROOT_PREFIX/micromamba"

    # test the mamba binary
    echo Micromamba version:
    "$MAMBA_ROOT_PREFIX/micromamba" --version

    # run the shell hook, otherwise activate will fail
    eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook -s posix)"

    # install git and python into the installer env
    if [ ! -e "$INSTALL_ENV_DIR" ]; then
        micromamba create -y --prefix "$INSTALL_ENV_DIR"
    fi

    micromamba install -y --prefix "$INSTALL_ENV_DIR" -c conda-forge $PACKAGES_TO_INSTALL

    # activate
    micromamba activate "$INSTALL_ENV_DIR"
fi

# get the repo (and load into the current directory)
if [ ! -e ".git" ]; then
    git init
    git remote add origin https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    git fetch
    git checkout origin/master -ft
fi

# run the script
bash webui.sh
