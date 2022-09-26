@echo off

set conda_env_name=automatic1111

:: Put the path to conda directory after "=" sign if it's installed at non-standard path:
set cutom_conda_path=D:\Software\Miniconda3

set paths=%ProgramData%\miniconda3
set paths=%paths%;%USERPROFILE%\miniconda3
set paths=%paths%;%ProgramData%\anaconda3
set paths=%paths%;%USERPROFILE%\anaconda3

IF NOT "%cutom_conda_path%"=="" (
  set paths=%cutom_conda_path%;%paths%
)

for %%a in (%paths%) do ( 
 if EXIST "%%a\Scripts\activate.bat" (
    SET CONDA_PATH=%%a
 )
)

IF "%CONDA_PATH%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  exit /b 1 
) else (
  echo anaconda3/miniconda3 detected in %CONDA_PATH%
)

call "%CONDA_PATH%\Scripts\activate.bat"
call "%CONDA_PATH%\Scripts\activate.bat" "%conda_env_name%"
python %CD%/launch.py

