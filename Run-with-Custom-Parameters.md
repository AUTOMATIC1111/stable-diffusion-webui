# webui-user
The recommended way to customize how the program is run is editing `webui-user.bat` (Windows) and `webui-user.sh` (Linux):
- `set PYTHON` allows for setting a custom Python path
    - Example: `set PYTHON=b:/soft/Python310/Python.exe`
- `set VENV_DIR` allows you to chooser the directory for the virtual environment. Default is `venv`. Special value `-` runs the script without creating virtual environment.
    - Example: `set VENV_DIR=C:\run\var\run` will create venv in the `C:\run\var\run` directory.
    - Example: `set VENV_DIR=-` runs the program using the system's python
- `set COMMANDLINE_ARGS` setting the command line arguments `webui.py` is ran with
    - Example: `set COMMANDLINE_ARGS=--ckpt a.ckpt` uses the model `a.ckpt` instead of `model.ckpt`

# Command Line Arguments
## Creating Large Images
Use `--opt-split-attention` parameter. It slows down sampling a tiny bit, but allows you to make gigantic images.

## Running online
Use the `--share` option to run online. You will get a xxx.app.gradio link. This is the intended way to use the program in collabs. You may set up authentication for said gradio shared instance with the flag `--gradio-auth username:password`, optionally providing multiple sets of usernames and passwords separated by commas.

Use `--listen` to make the server listen to network connections. This will allow computers on the local network to access the UI, and if you configure port forwarding, also computers on the internet.

Use `--port xxxx` to make the server listen on a specific port, xxxx being the wanted port. Remember that all ports below 1024 need root/admin rights, for this reason it is advised to use a port above 1024. Defaults to port 7860 if available.
