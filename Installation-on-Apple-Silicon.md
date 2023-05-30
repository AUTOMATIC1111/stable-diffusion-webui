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
4. Place Stable Diffusion models/checkpoints you want to use into `stable-diffusion-webui/models/Stable-diffusion`. If you don't have any, see [Downloading Stable Diffusion Models](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon#downloading-stable-diffusion-models) below.
5. `cd stable-diffusion-webui` and then `./webui.sh` to run the web UI. A Python virtual environment will be created and activated using venv and any remaining missing dependencies will be automatically downloaded and installed.
6. To relaunch the web UI process later, run `./webui.sh` again. Note that it doesn't auto update the web UI; to update, run `git pull` before running `./webui.sh`.

### Existing Install:
If you have an existing install of web UI that was created with `setup_mac.sh`, delete the `run_webui_mac.sh` file and `repositories` folder from your `stable-diffusion-webui` folder. Then run `git pull` to update web UI and then `./webui.sh` to run it.

Downloading Stable Diffusion Models
------

If you don't have any models to use, Stable Diffusion models can be downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&sort=downloads). To download, click on a model and then click on the `Files and versions` header. Look for files listed with the ".ckpt" or ".safetensors" extensions, and then click the down arrow to the right of the file size to download them.

Some popular official Stable Diffusion models are:
* [Stable DIffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) ([sd-v1-4.ckpt](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt))
* [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) ([v1-5-pruned-emaonly.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt))
* [Stable Diffusion 1.5 Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) ([sd-v1-5-inpainting.ckpt](https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt))

Stable Diffusion 2.0 and 2.1 require both a model and a configuration file, and image width & height will need to be set to 768 or higher when generating images:
* [Stable Diffusion 2.0](https://huggingface.co/stabilityai/stable-diffusion-2) ([768-v-ema.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt))
* [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) ([v2-1_768-ema-pruned.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt))

For the configuration file, hold down the option key on the keyboard and click [here](https://github.com/Stability-AI/stablediffusion/raw/main/configs/stable-diffusion/v2-inference-v.yaml) to download `v2-inference-v.yaml` (it may download as `v2-inference-v.yaml.yml`). In Finder select that file then go to the menu and select `File` > `Get Info`. In the window that appears select the filename and change it to the filename of the model, except with the file extension `.yaml` instead of `.ckpt`, press return on the keyboard (confirm changing the file extension if prompted), and place it in the same folder as the model (e.g. if you downloaded the `768-v-ema.ckpt` model, rename it to `768-v-ema.yaml` and put it in `stable-diffusion-webui/models/Stable-diffusion` along with the model).

Also available is a [Stable Diffusion 2.0 depth model](https://huggingface.co/stabilityai/stable-diffusion-2-depth) ([512-depth-ema.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/512-depth-ema.ckpt)). Download the `v2-midas-inference.yaml` configuration file by holding down option on the keyboard and clicking [here](https://github.com/Stability-AI/stablediffusion/raw/main/configs/stable-diffusion/v2-midas-inference.yaml), then rename it with the `.yaml` extension in the same way as mentioned above and put it in `stable-diffusion-webui/models/Stable-diffusion` along with the model. Note that this model works at image dimensions of 512 width/height or higher instead of 768.

Troubleshooting
------

### Web UI Won't Start:
If you encounter errors when trying to start the Web UI with `./webui.sh`, try deleting the `repositories` and `venv` folders from your `stable-diffusion-webui` folder and then update web UI with `git pull` before running `./webui.sh` again.

### Poor Performance:
Currently GPU acceleration on macOS uses a _lot_ of memory. If performance is poor (if it takes more than a minute to generate a 512x512 image with 20 steps with any sampler) 
- Try starting with the `--opt-split-attention-v1` command line option (i.e. `./webui.sh --opt-split-attention-v1`) and see if that helps. 
- Doesn't make much difference?
  - Open the Activity Monitor application located in /Applications/Utilities and check the memory pressure graph under the Memory tab. Memory pressure is being displayed in red when an image is generated
  - Close the web UI process and then add the `--medvram` command line option (i.e. `./webui.sh --opt-split-attention-v1 --medvram`). 
- Performance is still poor and memory pressure still red with that option?
  - Try `--lowvram` (i.e. `./webui.sh --opt-split-attention-v1 --lowvram`). 
- Still takes more than a few minutes to generate a 512x512 image with 20 steps with with any sampler?
  - You may need to turn off GPU acceleration. 
    - Open `webui-user.sh` in Xcode 
    - Change `#export COMMANDLINE_ARGS=""` to `export COMMANDLINE_ARGS="--skip-torch-cuda-test --no-half --use-cpu all"`.

------

Discussions/Feedback here: https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5461