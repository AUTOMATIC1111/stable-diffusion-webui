#!/usr/bin/env bash -l

# This should not be needed since it's configured during installation, but might as well have it here.
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate conda environment
conda activate web-ui

# Pull the latest changes from the repo
git pull --rebase

# Run the web ui
python webui.py --precision full --no-half --use-cpu Interrogate GFPGAN CodeFormer BSRGAN ESRGAN SCUNet $@

# Deactivate conda environment
conda deactivate

