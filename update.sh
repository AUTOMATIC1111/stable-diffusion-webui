#!/bin/bash

INSTALL_ENV_DIR=$(pwd)/installer_files/env
export PATH=$PATH;$INSTALL_ENV_DIR/bin

# update the repo
if [ -e ".git" ]; then
    git pull
fi
