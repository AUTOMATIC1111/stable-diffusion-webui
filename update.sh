#!/bin/bash

INSTALL_ENV_DIR="$(pwd)/installer_files/env"

if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$INSTALL_ENV_DIR/bin:$PATH"; fi

# update the repo
if [ -e ".git" ]; then
    git pull
fi
