@echo off

set PYTHON=C:\ProgramData\Anaconda3\envs\image_generation\python.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--theme dark
::set COMMANDLINE_ARGS=--theme dark --medvram
::set COMMANDLINE_ARGS=--theme dark --lowvram
::set COMMANDLINE_ARGS=--theme dark --use-cpu all --no-half


call webui.bat
