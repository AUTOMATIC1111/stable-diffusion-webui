@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--autolaunch --xformers --theme dark --opt-channelslast --vae-path "D:\AI\automatic\models\Stable-diffusion\newVAE.vae.pt" --medvram
call webui.bat
