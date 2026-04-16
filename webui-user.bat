@echo off

set PYTHON=C:\Users\nicol\AppData\Local\Programs\Python\Python310\python.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api
set STABLE_DIFFUSION_REPO=https://github.com/CompVis/stable-diffusion.git
set STABLE_DIFFUSION_COMMIT_HASH=21f890f9da3cfbeaba8e2ac3c425ee9e998d5229

call webui.bat
