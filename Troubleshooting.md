- **The program is tested to work on Python 3.10.6. Don't use other versions unless you are looking for trouble.**
- The installer creates a python virtual environment, so none of the installed modules will affect existing system installations of python.
- To use the system's python rather than creating a virtual environment, use custom parameter replacing `set VENV_DIR=-`.
- To reinstall from scratch, delete directories: `venv`, `repositories`.
- When starting the program for the first time, the path to python interpreter is displayed. If this is not the python you installed, you can specify full path in the `webui-user` script; see [Running with custom parameters](Run-with-Custom-Parameters).
- If the desired version of Python is not in PATH, modify the line `set PYTHON=python` in `webui-user.bat` with the full path to the python executable.
    - Example: `set PYTHON=B:\soft\Python310\python.exe`
- Installer requirements from `requirements_versions.txt`, which lists versions for modules specifically compatible with Python 3.10.6. If this doesn't work with other versions of Python, setting the custom parameter `set REQS_FILE=requirements.txt` may help.

# Low VRAM Video-cards
When running on video cards with a low amount of VRAM (<=4GB), out of memory errors may arise.
Various optimizations may be enabled through command line arguments, sacrificing some/a lot of speed in favor of using less VRAM:
- If you have 4GB VRAM and want to make 512x512 (or maybe up to 640x640) images, use `--medvram`.
- If you have 4GB VRAM and want to make 512x512 images, but you get an out of memory error with `--medvram`, use `--lowvram --always-batch-cond-uncond` instead.
- If you have 4GB VRAM and want to make images larger than you can with `--medvram`, use  `--lowvram`.

# Green or Black screen
Video cards
When running on video cards which don't support half precision floating point numbers (a known issue with 16xx cards), a green or black screen may appear instead of the generated pictures.
This may be fixed by using the command line arguments `--precision full --no-half` at a significant increase in VRAM usage, which may require `--medvram`.

# "CUDA error: no kernel image is available for execution on the device" after enabling xformers
Your installed xformers is incompatible with your GPU. If you use Python 3.10, have a Pascal or higher card and run on Windows, add `--reinstall-xformers --xformers` to your `COMMANDLINE_ARGS` to upgrade to a working version. Remove `--reinstall-xformers` after upgrading.

# NameError: name 'xformers' is not defined
If you use Windows, this means your Python is too old. Use 3.10

If Linux, you'll have to build xformers yourself or just avoid using xformers.