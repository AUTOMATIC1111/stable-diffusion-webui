@echo off

set PYTHON=C:\Users\moham\AppData\Local\Programs\Python\Python310\python.exe
set COMMANDLINE_ARGS=--skip-torch-cuda-test

call webui.bat %COMMANDLINE_ARGS%