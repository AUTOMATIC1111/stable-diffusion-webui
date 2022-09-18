- The program is tested to work on 3.10.6. Don't use other versions unless you are looking for trouble.
- The installer creates a python virtual environment, so none of the installed modules will affect existing system installations of python.
- To use the system's python rather than creating a virtual environment, use custom parameter replacing `set VENV_DIR=-`.
- To reinstall from scratch, delete directories: `venv`, `repositories`.
- When starting the program for the first time, the path to python interpreter is displayed. If this is not the python you installed, you can specify full path in the `webui-user` script; see [Running with custom parameters](Run-with-Custom-Parameters).

# Low VRAM Video-cards
When running on video cards with a low amount of VRAM (<=4GB), out of memory errors may arise.
Various optimizations may be enabled through command line arguments, sacrificing some/a lot of speed in favor of using less VRAM:
- If you have 4GB VRAM and want to make 512x512 (or maybe up to 640x640) images, use `--medvram`.
- If you have 4GB VRAM and want to make 512x512 images, but you get an out of memory error with `--medvram`, use `--medvram --opt-split-attention` instead.
- If you have 4GB VRAM and want to make 512x512 images, and you still get an out of memory error, use `--lowvram --always-batch-cond-uncond --opt-split-attention` instead.
- If you have 4GB VRAM and want to make images larger than you can with `--medvram`, use  `--lowvram --opt-split-attention`.
- If you have more VRAM and want to make larger images than you can usually make (for example 1024x1024 instead of 512x512), use `--medvram --opt-split-attention`. You can use `--lowvram` also but the effect will likely be barely noticeable.
- Otherwise, do not use any of those.

# Green or Black screen
Video cards
When running on video cards which don't support half precision floating point numbers (a known issue with 16xx cards), a green or black screen may appear instead of the generated pictures.
This may be fixed by using the command line arguments `--precision full --no-half` at a significant increase in VRAM usage, which may require `--medvram`.
