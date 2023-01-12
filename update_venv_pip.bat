set VENV_DIR=venv
set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"
%PYTHON% -m pip list %*
pause
%PYTHON% -m pip install --upgrade pip %*
pause