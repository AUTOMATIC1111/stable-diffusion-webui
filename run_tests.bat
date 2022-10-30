@echo off
set ERROR_REPORTING=FALSE
set COMMANDLINE_ARGS= --api
echo Launching SDWebUI...
start "SDWebUITest" webui.bat

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set VENV_DIR=venv)
set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"
%PYTHON% test/server_poll.py
for /f "tokens=2 delims=," %%a in ('tasklist /v /fo csv ^| findstr /i "SDWebUITest"') do set "$PID=%%a"

taskkill /PID %$PID% >nul 2>&1

pause
