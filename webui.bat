@echo off

set PYTHON=python
set GIT=git
set COMMANDLINE_ARGS=

mkdir tmp 2>NUL

set TORCH_COMMAND=pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_git
echo Couldn't launch python
goto :show_stdout_stderr

:check_git
%GIT% --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_torch
echo Couldn't launch git
goto :show_stdout_stderr

:install_torch
%PYTHON% -c "import torch" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_gpu
echo Installing torch...
%PYTHON% -m %TORCH_COMMAND% >tmp/stdout.txt 2>tmp/stderr.txt

if %ERRORLEVEL% == 0 goto :check_gpu
echo Failed to install torch
goto :show_stdout_stderr

:check_gpu
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_sd_reqs
echo Torch is not able to use GPU
goto :show_stdout_stderr

:install_sd_reqs
%PYTHON% -c "import transformers" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_k_diff
echo Installing SD requirements...
%PYTHON% -m pip install transformers==4.19.2 diffusers invisible-watermark >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_k_diff
goto :show_stdout_stderr

:install_k_diff
%PYTHON% -c "import k_diffusion.sampling" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_GFPGAN
echo Installing K-Diffusion...
%PYTHON% -m pip install git+https://github.com/crowsonkb/k-diffusion.git >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_GFPGAN
goto :show_stdout_stderr


:install_GFPGAN
%PYTHON% -c "import gfpgan" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_reqs
echo Installing GFPGAN
%PYTHON% -m pip install git+https://github.com/TencentARC/GFPGAN.git >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install_reqs
goto :show_stdout_stderr

:install_reqs
%PYTHON% -c "import omegaconf" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :make_dirs
echo Installing requirements...
%PYTHON% -m pip install -r requirements.txt >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :update_numpy
goto :show_stdout_stderr
:update_numpy
%PYTHON% -m pip install -U numpy >tmp/stdout.txt 2>tmp/stderr.txt

:make_dirs
mkdir repositories 2>NUL

if exist repositories\stable-diffusion goto :clone_transformers
echo Cloning Stable Difusion repository...
%GIT% clone https://github.com/CompVis/stable-diffusion.git repositories\stable-diffusion >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :clone_transformers
goto :show_stdout_stderr

:clone_transformers
if exist repositories\taming-transformers goto :check_model
echo Cloning Taming Transforming repository...
%GIT% clone https://github.com/CompVis/taming-transformers.git repositories\taming-transformers >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_model
goto :show_stdout_stderr

:check_model
dir model.ckpt >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_gfpgan
echo Stable Diffusin model not found: you need to place model.ckpt file into same directory as this file.
goto :show_stdout_stderr

:check_gfpgan
dir GFPGANv1.3.pth >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo GFPGAN not found: you need to place GFPGANv1.3.pth file into same directory as this file.
echo Face fixing feature will not work.

:launch
echo Launching webui.py...
cd repositories\stable-diffusion
%PYTHON% ..\..\webui.py %COMMANDLINE_ARGS%
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
echo Launch unsuccessful. Exiting.
