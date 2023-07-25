@echo off

git pull upstream master

set SD_ROOT=S:/StableDiffusion/data

set PYTHON=python3.10.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=^
 --enable-insecure-extension-access^
 --listen^
 --xformers^
 --theme=dark^
 --bsrgan-models-path=%SD_ROOT%/bsrgan^
 --ckpt-dir=%SD_ROOT%/checkpoints^
 --clip-models-path=%SD_ROOT%/clip^
 --codeformer-models-path=%SD_ROOT%/codeformer^
 --embeddings-dir=%SD_ROOT%/embeddings^
 --esrgan-models-path=%SD_ROOT%/esrgan^
 --gfpgan-dir=%SD_ROOT%/gfpgan^
 --gfpgan-models-path=%SD_ROOT%/gfpgan^
 --ldsr-models-path=%SD_ROOT%/ldsr^
 --lora-dir=%SD_ROOT%/lora^
 --realesrgan-models-path=%SD_ROOT%/realesrgan^
 --scunet-models-path=%SD_ROOT%/scunet^
 --swinir-models-path=%SD_ROOT%/swinir^
 --vae-dir=%SD_ROOT%/vae

call webui.bat