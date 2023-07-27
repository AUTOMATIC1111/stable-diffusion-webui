@echo off

git pull upstream master

set HF_DATASETS_CACHE=S:/HuggingFace/.cache/datasets
set SD_DATA=S:/StableDiffusion/data

set PYTHON=python3.10.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=^
 --enable-insecure-extension-access^
 --listen^
 --xformers^
 --theme=dark^
 --bsrgan-models-path=%SD_DATA%/bsrgan^
 --ckpt-dir=%SD_DATA%/checkpoints^
 --clip-models-path=%SD_DATA%/clip^
 --codeformer-models-path=%SD_DATA%/codeformer^
 --embeddings-dir=%SD_DATA%/embeddings^
 --esrgan-models-path=%SD_DATA%/esrgan^
 --gfpgan-dir=%SD_DATA%/gfpgan^
 --gfpgan-models-path=%SD_DATA%/gfpgan^
 --ldsr-models-path=%SD_DATA%/ldsr^
 --lora-dir=%SD_DATA%/lora^
 --realesrgan-models-path=%SD_DATA%/realesrgan^
 --scunet-models-path=%SD_DATA%/scunet^
 --swinir-models-path=%SD_DATA%/swinir^
 --vae-dir=%SD_DATA%/vae

call webui.bat