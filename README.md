# Stable Diffusion web UI
A browser interface based on Gradio library for Stable Diffusion.

![](screenshot.png)

## Features
[Detailed feature showcase with images](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Prompt matrix
- Stable Diffusion upscale
- Attention
- Loopback
- X/Y plot
- Textual Inversion
- Extras tab with:
    - GFPGAN, neural network that fixes faces
    - CodeFormer, face restoration tool as an alternative to GFPGAN
    - RealESRGAN, neural network upscaler
    - ESRGAN, neural network with a lot of third party models
- Resizing aspect ratio options
- Sampling method selection
- Interrupt processing at any time
- 4GB video card support
- Correct seeds for batches
- Prompt length validation
- Generation parameters added as text to PNG
- Tab to view an existing picture's generation parameters
- Settings page
- Running custom code from UI
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Random artist button
- Tiling support: UI checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
- Negative prompt
- Styles
- Variations
- Seed resizing
- CLIP interrogator
- Prompt Editing

## Installation and Running
Make sure the required [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) are met and follow the instructions available for both [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended) and [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.

Alternatively, use [Google Colab](https://colab.research.google.com/drive/1Iy-xW9t1-OQWhb0hNxueGij8phCyluOh).

### Automatic Installation on Windows
1. Install [Python 3.10.6](https://www.python.org/downloads/windows/), checking "Add Python to PATH"
2. Install [git](https://git-scm.com/download/win).
3. Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Place `model.ckpt` in the `models` directory (see [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) for where to get it).
5. _*(Optional)*_ Place `GFPGANv1.4.pth` in the base directory, alongside `webui.py` (see [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) for where to get it).
6. Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install wget git python3 python3-venv
# Red Hat-based:
sudo dnf install wget git python3
# Arch-based:
sudo pacman -S wget git python3
```
2. To install in `/home/$(whoami)/stable-diffusion-webui/`, run:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```

### Installation on Apple Silicon (unstable)

**IMPORTANT** While the web UI runs fine, there are still certain issues when running this fork on Apple Silicon.
The only 2 samplers that work (at the time of writing this) are `Euler` and `DPM2` - all others result in a black screen.
Upscaling works, but only using the real-ESRGAN models.

First get the weights checkpoint download started - it's big:

Sign up at https://huggingface.co
Go to the Stable diffusion diffusion model page
Accept the terms and click Access Repository:
Download sd-v1-4.ckpt (4.27 GB) and note where you have saved it (probably the Downloads folder)

1. `brew install cmake protobuf rust`
2. `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o Miniconda3-latest-MacOSX-arm64.sh`
3. `/bin/bash Miniconda3-latest-MacOSX-arm64.sh`
4. `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
5. `cd stable-diffusion-webui`
6. Run the following commands one by one: 
```
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
 
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers

git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    
git clone https://github.com/salesforce/BLIP.git repositories/BLIP
```

7. `conda create --name web_ui python=3.10`
8. `conda activate web_ui`
9. `pip install -r requirements.txt`
10. `conda install pytorch torchvision torchaudio -c pytorch-nightly`
11. At this point, move the downloaded `sd-v1-4.ckpt` file into `stable-diffusion-webui/models/`. You will know it's the right folder since there's a text file named `Put Stable Diffusion checkpoints here.txt` in it.
12. `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1`
13. `python webui.py --precision full --no-half --opt-split-attention-v1`

It is possible that after running `webui.py` you get error messages saying certain packages are missing. Install the missing package, go back to step 13 and repeat.

#### Common Errors

##### Error

`ImportError: dlopen(.venv/lib/python3.10/site-packages/google/protobuf/pyext/_message.cpython-310-darwin.so, 0x0002): symbol not found in flat namespace (__ZN6google8protobuf15FieldDescriptor12TypeOnceInitEPKS1_)`

##### Solution

Downgrade Protobuf using `pip install protobuf==3.19.4`




## Documentation
The documentation was moved from this README over to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Doggettx - Cross Attention layer optimization - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
