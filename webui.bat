:: --------------------------------------------------------------------------------------------------------------
:: Do not make any changes to this file. Instead, create a shortcut to this file and add needed arguments there.
:: --------------------------------------------------------------------------------------------------------------

@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
set ERROR_REPORTING=FALSE
mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Cannot launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :start_venv
echo Cannot install pip
goto :show_stdout_stderr

:start_venv
if ["%VENV_DIR%"] == ["-"] goto :skip_venv
if ["%SKIP_VENV%"] == ["1"] goto :skip_venv

dir "%VENV_DIR%\Scripts\Python.exe" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Using python: %PYTHON_FULLNAME%
echo Creating VENV: %VENV_DIR%
%PYTHON_FULLNAME% -m venv "%VENV_DIR%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :activate_venv
echo Failed creating VENV: "%VENV_DIR%"
goto :show_stdout_stderr

:activate_venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo Using VENV: %VENV_DIR%

:skip_venv
if [%ACCELERATE%] == ["True"] goto :accelerate
goto :launch

:accelerate
set ACCELERATE="%VENV_DIR%\Scripts\accelerate.exe"
if EXIST %ACCELERATE% goto :accelerate_launch

:launch
%PYTHON% launch.py %*
pause
exit /b

:accelerate_launch
echo Using accelerate
%ACCELERATE% launch --num_cpu_threads_per_process=6 launch.py %*
pause
exit /b

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
echo Launch Failed
pause
