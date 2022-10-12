@echo off

@rem This script will install git and python (if not found on the PATH variable)
@rem  using micromamba (an 8mb static-linked single-file binary, conda replacement).
@rem For users who already have git and python, this step will be skipped.

@rem Next, it'll checkout the project's git repo, if necessary.
@rem Finally, it'll start webui-user.bat (to proceed with the usual installation).

@rem This enables a user to install this project without manually installing python and git.

@rem prevent the window from closing after an error
if not defined in_subprocess (cmd /k set in_subprocess=y ^& %0 %*) & exit )

@rem config
set MAMBA_ROOT_PREFIX=%cd%\installer_files\mamba
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MICROMAMBA_BINARY_FILE=%cd%\installer_files\micromamba_win_x64.exe
set PATH=%PATH%;%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts

@rem figure out what needs to be installed
set PACKAGES_TO_INSTALL=

call python --version "" >tmp/stdout.txt 2>tmp/stderr.txt
if "%ERRORLEVEL%" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% python

call git --version "" >tmp/stdout.txt 2>tmp/stderr.txt
if "%ERRORLEVEL%" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% git

@rem install git and python into a contained environment (if necessary)
if "%PACKAGES_TO_INSTALL%" NEQ "" (
    echo "Packages to install: %PACKAGES_TO_INSTALL%"

    @rem initialize the mamba binary
    if not exist "%MAMBA_ROOT_PREFIX%" mkdir "%MAMBA_ROOT_PREFIX%"
    copy "%MICROMAMBA_BINARY_FILE%" "%MAMBA_ROOT_PREFIX%\micromamba.exe"

    @rem test the mamba binary
    echo Micromamba version:
    call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version

    @rem run the shell hook, otherwise activate will fail
    if not exist "%MAMBA_ROOT_PREFIX%\Scripts" (
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" shell hook --log-level 4 -s cmd.exe
    )

    call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"

    @rem install git and python into the installer env
    if not exist "%INSTALL_ENV_DIR%" (
        call micromamba create -y --prefix "%INSTALL_ENV_DIR%"
    )

    call micromamba install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge %PACKAGES_TO_INSTALL%

    @rem activate
    call micromamba activate "%INSTALL_ENV_DIR%"
)

@rem get the repo (and load into the current directory)
if not exist ".git" (
    call git init
    call git remote add origin https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    call git fetch
    call git checkout origin/master -ft
)

@rem run the script
call webui-user.bat

pause
