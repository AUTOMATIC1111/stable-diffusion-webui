@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

echo AUTOMATIC1111 / stable-diffusion-webui
CHOICE /T 5 /M "Do you want to check for Updates? Yes[y] or No[n]?" /D N
IF ERRORLEVEL 2 (GOTO :launch)
    
:update
echo Checking for Updates...
git pull https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

:launch
cls
call webui.bat
