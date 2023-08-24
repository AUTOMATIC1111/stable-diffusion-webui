@echo off

:: The local stable-diffusion-webui folder is a fork of the AUTOMATIC1111 repo.
:: A "remote" has been added called "A1111" that points at the original repo so
:: we pull the latest changes into the forked one when starting the webui:
git pull A1111 master

:: A (futile?) attempt to move massive files off the C drive:
set HF_DATASETS_CACHE=S:\\HuggingFace\\.cache\\datasets
set TRANSFORMERS_CACHE=S:\\HuggingFace\\.cache\\transformers

set PYTHON=python3.10.exe
set GIT=
set VENV_DIR=

set SD_DATA=S:\\StableDiffusion\\data
set COMMANDLINE_ARGS=^
 --data-dir=%SD_DATA%^
 --ckpt-dir=%SD_DATA%\\models\\Checkpoints^
 --vae-dir=%SD_DATA%\\models\\VAE^
 --enable-insecure-extension-access^
 --listen^
 --no-half-vae^
 --theme=dark^
 --xformers

:: data-dir argument
::
:: This sets the root path for the following folders. Useful to avoid losing
:: large downloads if you delete the main stable-diffusion-webui folder.
::
:: + extensions
:: + models
::   + Codeformer
::   + ControlNet
::   + ESRGAN
::   + GFPGAN
::   + hypernetworks
::   + LDSR
::   + Lora
::   + Stable-diffusion
::   + SwinIR

call webui.bat