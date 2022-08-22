# Stable Diffusion web UI
A browser interface based on Gradio library for Stable Diffusion.

Original script with Gradio UI was written by a kind anonymopus user. This is a modification.

![](screenshot.png)

## Stable Diffusion

This script assumes that you already have main Stable Diffusion sutff installed, assumed to be in directory `/sd`.
If you don't have it installed, follow the guide:

- https://rentry.org/kretard

This repository's `webgui.py` is a replacement for `kdiff.py` from the guide.

Particularly, following files must exist:

- `/sd/configs/stable-diffusion/v1-inference.yaml`
- `/sd/models/ldm/stable-diffusion-v1/model.ckpt`
- `/sd/ldm/util.py`
- `/sd/k_diffusion/__init__.py`

## GFPGAN

If you want to use GFPGAN to improve generated faces, you need to install it separately.
Follow instructions from https://github.com/TencentARC/GFPGAN, but when cloning it, do so into Stable Diffusion main directory, `/sd`.
After that download [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) and put it
into the `/sd/GFPGAN/experiments/pretrained_models` directory. If you're getting troubles with GFPGAN support, follow instructions
from the GFPGAN's repository until `inference_gfpgan.py` script works.

The following files must exist:

- `/sd/GFPGAN/inference_gfpgan.py`
- `/sd/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth`

If the GFPGAN directory does not exist, you will not get the option to use GFPGAN in the UI. If it does exist, you will either be able
to use it, or there will be a message in console with an error related to GFPGAN.

## Web UI

Run the script as:

`python webui.py`

When running the script, you must be in the main Stable Diffusion directory, `/sd`. If you cloned this repository into a subdirectory 
of `/sd`, say, the `stable-diffusion-webui` directory, you will run it as:

`python stable-diffusion-webui/webui.py`

When launching, you may get a very long warning message related to some weights not being used. You may freely ignore it.
After a while, you will get a message like this:

```
Running on local URL:  http://127.0.0.1:7860/
```

Open the URL in browser, and you are good to go.
