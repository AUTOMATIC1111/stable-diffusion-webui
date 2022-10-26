## Important notes

 While the web UI runs fine, there are still certain issues when running this fork on Apple Silicon.
The only 2 samplers that work (at the time of writing this) are `Euler` and `DPM2` - all others result in a black screen.
Upscaling works, but only using the real-ESRGAN models.

EDIT: They all seem to be working now except for "DPM fast" (which returns random noise), and DDIM and PLMS (both of which fail immediately with the following error: "AssertionError: Torch not compiled with CUDA enabled"). According to a user on an M1 Max macbook pro with 32 gigs of ram https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/3694

People have also had issues completing the installation because they have previous installations of Python and/or Miniconda. This causes the script to be confused about where to install the environment and all related files. 

Please follow [this guide](https://apple.stackexchange.com/questions/284824/remove-and-reinstall-python-on-mac-can-i-trust-these-old-references/285028#285028) if this is the case.

## Automatic installation

First, you need to install the required dependencies using [Homebrew](https://brew.sh).

`brew install cmake protobuf rust python git wget`

The script can be downloaded from [here](https://github.com/dylancl/stable-diffusion-webui-mps/blob/master/setup_mac.sh), or follow the instructions below.

1. Open Terminal.app
2. Run the following commands:

```
$ cd ~/Documents/
$ curl https://raw.githubusercontent.com/dylancl/stable-diffusion-webui-mps/master/setup_mac.sh -o setup_mac.sh
$ chmod +x setup_mac.sh
$ ./setup_mac.sh
```

3. Follow the instructions in the terminal window.

#### Usage

After installation, you'll now find `run_webui_mac.sh` in the `stable-diffusion-webui` directory. Run this script to start the web UI using `./run_webui_mac.sh`.
This script automatically activates the conda environment, pulls the latest changes from the repository, and starts the web UI. On exit, the conda environment is deactivated.