@echo off

set INSTALL_ENV_DIR=%cd%\installer_files\env
set PATH=%PATH%;%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts

@rem update the repo
if exist ".git" (
    call git pull
)

pause