@echo off

set PYTHON=
@REM The "Nuullll/intel-extension-for-pytorch" wheels were built from IPEX source for Intel Arc GPU: https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main
@REM This is NOT an Intel official release so please use it at your own risk!!
@REM See https://github.com/Nuullll/intel-extension-for-pytorch/releases/tag/v2.0.110%2Bxpu-master%2Bdll-bundle for details.
@REM 
@REM Strengths (over official IPEX 2.0.110 windows release):
@REM   - AOT build (for Arc GPU only) to eliminate JIT compilation overhead: https://github.com/intel/intel-extension-for-pytorch/issues/399
@REM   - Bundles minimal oneAPI 2023.2 dependencies into the python wheels, so users don't need to install oneAPI for the whole system.
@REM   - Provides a compatible torchvision wheel: https://github.com/intel/intel-extension-for-pytorch/issues/465
@REM Limitation:
@REM   - Only works for python 3.10
set "TORCH_COMMAND=pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.0.110%%2Bxpu-master%%2Bdll-bundle/torch-2.0.0a0+gite9ebda2-cp310-cp310-win_amd64.whl https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.0.110%%2Bxpu-master%%2Bdll-bundle/torchvision-0.15.2a0+fa99a53-cp310-cp310-win_amd64.whl https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.0.110%%2Bxpu-master%%2Bdll-bundle/intel_extension_for_pytorch-2.0.110+gitc6ea20b-cp310-cp310-win_amd64.whl"
set GIT=
set VENV_DIR=
set "COMMANDLINE_ARGS=--use-ipex --skip-torch-cuda-test --skip-version-check --opt-sdp-attention"

call webui.bat
