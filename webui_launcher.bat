@echo off
title Stable Diffusion

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set VENV_DIR=venv)

:: Check if python is installed
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :main
echo Couldn't launch python
goto :show_stdout_stderr

:main
    :: get python version
    for /f "tokens=2 delims= " %%a in ('%PYTHON% -V 2^>^&1') do set PYTHON_VERSION=%%a

    call :check_python_version

    call :setup_venv

    goto :run_python_script
    exit 0


:check_python_version
    if %PYTHON_VERSION:~0,1% LSS 3 (
        call :old_python_version >tmp/stdout.txt 2>tmp/stderr.txt
    )
    if %PYTHON_VERSION:~2,1% EQU 1 (
        if %PYTHON_VERSION:~3,1% LSS 0 (
            call :old_python_version >tmp/stdout.txt 2>tmp/stderr.txt
        )
    ) else (
        call :old_python_version >tmp/stdout.txt 2>tmp/stderr.txt
    )
    if %ERRORLEVEL% == 0 exit /b 0
    goto :show_stdout_stderr


:old_python_version
    echo Python version %PYTHON_VERSION% is too old. Please install Python 3.10.6 or newer. 1>&2
    exit /b 1

:setup_venv
    if [%VENV_DIR%] == [-] exit /b 0

    :: Check if venv already exists
    dir %VENV_DIR%\Scripts\Python.exe >tmp/stdout.txt 2>tmp/stderr.txt
    if %ERRORLEVEL% == 0 (
        call :activate_venv
        %PYTHON% --version >tmp/stdout.txt 2>tmp/stderr.txt
        exit /b %ERRORLEVEL%
    )

    :: Create venv
    for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
    echo Creating venv in directory "%VENV_DIR%" using python %PYTHON_FULLNAME%
    %PYTHON_FULLNAME% -m venv %VENV_DIR%
    if %ERRORLEVEL% == 0 (
        echo Virtual environment created successfully.
        call :activate_venv
        %PYTHON% --version >tmp/stdout.txt 2>tmp/stderr.txt
        exit /b %ERRORLEVEL%
    )
    :: Venv creation failed
    echo Unable to create venv in directory %VENV_DIR%
    goto :show_stdout_stderr

:activate_venv
    set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"
    exit /b %ERRORLEVEL%

:run_python_script
    %PYTHON% -m scripts.webui_launcher
    if %ERRORLEVEL% == 0 exit /b 0
    exit /b 1

:show_stdout_stderr
    echo.
    echo exit code: %errorlevel%

    for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
    if %size% equ 0 goto :show_stderr
    echo.
    echo stdout:
    type tmp\stdout.txt

    :show_stderr
    for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
    if %size% equ 0 goto :show_stderr
    echo.
    echo stderr:
    type tmp\stderr.txt

    :endofscript

    echo.
    echo Launch unsuccessful. Exiting.
    pause
    exit /b 1