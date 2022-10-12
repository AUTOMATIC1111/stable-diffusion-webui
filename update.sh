#!/bin/bash

INSTALL_ENV_DIR="$(pwd)/installer_files/env"

if [ -e "$INSTALL_ENV_DIR" ]; then export PATH="$PATH;$INSTALL_ENV_DIR/bin"; fi

# update the repo
if [ -e ".git" ]; then
    git pull
fi
