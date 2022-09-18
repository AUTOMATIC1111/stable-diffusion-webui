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
## Running online
Use the `--share` option to run online. You will get a xxx.app.gradio link. This is the intended way to use the program in collabs. You may set up authentication for said gradio shared instance with the flag `--gradio-auth username:password`, optionally providing multiple sets of usernames and passwords separated by commas.

Use `--listen` to make the server listen to network connections. This will allow computers on the local network to access the UI, and if you configure port forwarding, also computers on the internet.

Use `--port xxxx` to make the server listen on a specific port, xxxx being the wanted port. Remember that all ports below 1024 need root/admin rights, for this reason it is advised to use a port above 1024. Defaults to port 7860 if available.

## All command line arguments
Below is the output of `python webui.py --help`. This list may be outdated - you can view help yourself by running that command.

```
  -h, --help            show this help message and exit
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of stable diffusion model; this checkpoint will be added to the list of checkpoints and loaded by default if you don't have a checkpoint selected in settings
  --ckpt-dir CKPT_DIR   path to directory with stable diffusion checkpoints
  --gfpgan-dir GFPGAN_DIR
                        GFPGAN directory
  --gfpgan-model GFPGAN_MODEL
                        GFPGAN model file name
  --no-half             do not switch the model to 16-bit floats
  --no-progressbar-hiding
                        do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser)
  --max-batch-count MAX_BATCH_COUNT
                        maximum batch count value for the UI
  --embeddings-dir EMBEDDINGS_DIR
                        embeddings directory for textual inversion (default: embeddings)
  --allow-code          allow custom script execution from webui
  --medvram             enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage
  --lowvram             enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage
  --always-batch-cond-uncond
                        disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
  --unload-gfpgan       does not do anything.
  --precision {full,autocast}
                        evaluate at this precision
  --share               use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)
  --esrgan-models-path ESRGAN_MODELS_PATH
                        path to directory with ESRGAN models
  --opt-split-attention
                        does not do anything
  --disable-opt-split-attention
                        disable an optimization that reduces vram usage by a lot
  --opt-split-attention-v1
                        enable older version of split attention optimization that does not consaumes all the VRAM it can find
  --listen              launch gradio with 0.0.0.0 as server name, allowing to respond to network requests
  --port PORT           launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available
  --show-negative-prompt
                        does not do anything
  --ui-config-file UI_CONFIG_FILE
                        filename to use for ui configuration
  --hide-ui-dir-config  hide directory configuration from webui
  --ui-settings-file UI_SETTINGS_FILE
                        filename to use for ui settings
  --gradio-debug        launch gradio with --debug option
  --gradio-auth GRADIO_AUTH
                        set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"
  --opt-channelslast    change memory type for stable diffusion to channels last
  --styles-file STYLES_FILE
                        filename to use for styles
  --autolaunch          open the webui URL in the system's default browser upon launch
```
