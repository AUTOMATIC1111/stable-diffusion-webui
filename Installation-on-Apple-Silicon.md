Mac users: Please provide feedback on if these instructions do or don't work for you, and if anything is unclear or you are otherwise still having problems with your install that are not currently [mentioned here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5461).

Important notes
------
Currently most functionality in the web UI works correctly on macOS, with the most notable exceptions being CLIP interrogator and training. Although training does seem to work, it is incredibly slow and consumes an excessive amount of memory. CLIP interrogator can be used but it doesn't work correctly with the GPU acceleration macOS uses so the default configuration will run it entirely via CPU (which is slow).

Most samplers are known to work with the only exception being the PLMS sampler when using the Stable Diffusion 2.0 model. Generated images with GPU acceleration on macOS should usually match or almost match generated images on CPU with the same settings and seed.

Automatic installation
------

### New install:
1. If Homebrew is not installed, follow the instructions at https://brew.sh to install it. Keep the terminal window open and follow the instructions under "Next steps" to add Homebrew to your PATH.
2. Open a new terminal window and run `brew install cmake protobuf rust python@3.10 git wget`
3. Clone the web UI repository by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
4. Copy any Stable Diffusion models you want to use to `stable-diffusion-webui/models/Stable-diffusion`
5. `cd stable-diffusion-webui` and then `./webui.sh` to run the web UI. A Python virtual environment will be created and activated using venv and any remaining missing dependencies will be automatically downloaded and installed.
6. To relaunch the web UI process later, run `./webui.sh` again. Note that it doesn't auto update the web UI; to update, run `git pull` before running `./webui.sh`.

### Existing Install:
If you have an existing install of web UI that was created with `setup_mac.sh`, delete the `run_webui_mac.sh` file and `repositories` folder from your `stable-diffusion-webui` folder. Then run `git pull` to update web UI and then `./webui.sh` to run it.

Troubleshooting
------

### Web UI Won't Start:
If you encounter errors when trying to start the Web UI with `./webui.sh`, try deleting the `repositories` and `venv` folders from your `stable-diffusion-webui` folder and then update web UI with `git pull` before running `./webui.sh` again.

### Poor Performance:
Currently GPU acceleration on macOS uses a _lot_ of memory. If performance is poor (if it takes more than a minute to generate a 512x512 image with 20 steps with any sampler), open the Activity Monitor application located in /Applications/Utilities and check the memory pressure graph under the Memory tab. If memory pressure is being displayed in red when an image is generated, close the web UI process and then start it with the `--medvram` command line option (i.e. `./webui.sh --medvram`). If performance is still poor and memory pressure still red with that option, then instead try `--lowvram` (i.e. `./webui.sh --lowvram`). If it still takes more than a few minutes to generate a 512x512 image with 20 steps with with any sampler, then you may need to turn off GPU acceleration. Open `webui-user.sh` in Xcode and change `#export COMMANDLINE_ARGS=""` to `export COMMANDLINE_ARGS="--skip-torch-cuda-test --no-half --use-cpu all"`.

------

Discussions/Feedback here: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5461