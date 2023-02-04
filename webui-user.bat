@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set torch_command = os.environ.get('TORCH_COMMAND', "pip install --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117")
set COMMANDLINE_ARGS= --xformers --no-half-vae --no-half --disable-nan-check --lowvram --use-cpu esrgan, interrogate --opt-sub-quad-attention  --administrator --gradio-img2img-tool color-sketch --enable-console-prompts --api --precision full --no-half
git pull
call webui.bat
