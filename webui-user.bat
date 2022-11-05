@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--allow-code --gradio-img2img-tool color-sketch --max-batch-count=100 --ui-config-file=ui-config.json --no-half-vae --vae-path=models/Stable-diffusion/Breast-Expansion/hyper_vae.pt --deepdanbooru

call webui.bat
