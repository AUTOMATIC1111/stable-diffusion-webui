@echo off
set PYTHON=D:\stable-diffusion-webui\stable-diffusion-webui\Python3.10\python.exe
set GIT=D:\stable-diffusion-webui\stable-diffusion-webui\Git\mingw64\libexec\git-core\git.exe
set VENV_DIR=
set COMMANDLINE_ARGS=--autolaunch --medvram --no-half --xformers --deepdanbooru --listen --port 7861 --enable-console-prompts
set GIT_PYTHON_REFRESH=quiet
set INSTALL_REQUIREMENTS=false
set GIT_CLONE=false
call webui.bat
