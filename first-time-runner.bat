@echo off

set "filePath=%cd%\webui-user.bat"


(
    echo @echo off
    echo.  
    echo set GIT=
    echo set VENV_DIR=
    echo set COMMANDLINE_ARGS=--skip-torch-cuda-test --precision full --no-half 
    echo set PYTORCH_TRACING_MODE=TORCHFX
    echo.   
    echo call webui.bat

) > %filepath%


call webui-user.bat

pause
