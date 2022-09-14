# Stable Diffusion Krita Plugin
A simple interface based on this repository: https://github.com/AUTOMATIC1111/stable-diffusion-webui

Requires Krita 5.1

## Usage example
[target.webm](https://user-images.githubusercontent.com/112324253/188291339-9d146a9a-ba9f-4671-9bd8-c8b55fd48ba6.webm)

## Updates
- webui.bat now starts both krita_server and webui on usual address: http://127.0.0.1:7860. Just don't try to run SD both in Krita and in webui simultaneously, it will give you CUDA error most likely.
- removed krita.bat and krita.sh because they were confusing people, and were pain to support. Just run webui.bat ***from this repository***.
- added CodeFormer support, it should work by default instead of GFPGAN. You can change it in the config tab.

## Installing and running

If you used previous version which used conda, please make a new install. Please install it separately from webui.
If you got any trouble after updating repo, please try to delete and reinstall it.

### Plugin installation

1. Open Krita and go into Settings - Manage Resources... - Open Resource Folder
2. Go into folder `pykrita` (create it if it doesn't exist)
3. Copy from this repository contents of folder `krita_plugin` into `pykrita` folder of your Krita. You should have `krita_diff` folder
   and `krita_diff.desktop` file in pykrita folder.
4. Restart Krita
5. Go into Settings - Configure Krita... - Python Plugin Manager
6. Activate plugin "Krita Stable Diffusion Plugin"
7. Restart Krita

### Server installation

You need [python](https://www.python.org/downloads/windows/) and [git](https://git-scm.com/download/win)
installed to run this, and an NVidia video card.

You need `model.ckpt`, Stable Diffusion model checkpoint, a big file containing the neural network weights. You
can obtain it from the following places:
 - [official download](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
 - [file storage](https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl)
 - magnet:?xt=urn:btih:3a4a612d75ed088ea542acac52f9f45987488d1c&dn=sd-v1-4.ckpt&tr=udp%3a%2f%2ftracker.openbittorrent.com%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337

You can optionally use GFPGAN to improve faces, to do so you'll need to download the model from [here](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) and place it in the same directory as `webui.bat`.

To use ESRGAN models, put them into ESRGAN directory in the same location as webui.py. A file will be loaded
as a model if it has .pth extension, and it will show up with its name in the UI. Grab models from the [Model Database](https://upscale.wiki/wiki/Model_Database).

> Note: RealESRGAN models are not ESRGAN models, they are not compatible. Do not download RealESRGAN models. Do not place
RealESRGAN into the directory with ESRGAN models. Thank you.

### Automatic installation/launch

- install [Python 3.10.6](https://www.python.org/downloads/windows/) and check "Add Python to PATH" during installation. You must install this exact version.
- install [git](https://git-scm.com/download/win)
- place `model.ckpt` into webui directory, next to `webui.bat`.
- _*(optional)*_ place `GFPGANv1.3.pth` into webui directory, next to `webui.bat`.
- run `webui-user.bat` from Windows Explorer. Run it as a normal user, ***not*** as administrator. You should run webui-user.bat ***from this repository***, not from others.

### Linux installation

Please look at parent repo, try to run webui.sh following instructions. You should run webui.sh ***from this repository***, not from others.

#### Troublehooting:

Look into parent repository https://github.com/AUTOMATIC1111/stable-diffusion-webui for instructions. This repository uses slightly changed code, but most parameters including those for low VRAM usage should still work.

### Usage

Put something in your prompt and just run it. If you select some area, only it will be used.

#### Hotkeys

- txt2img - Ctrl + Alt + Q
- img2img - Ctrl + Alt + W
- sd upscale - Ctrl + Alt + E
- inpainting - Ctrl + Alt + R
- just upscale, no SD - Ctrl + Alt + T

#### Img2img

You may use feathered selections. Selection will be converted to transparency mask afterwards. Not sure it is useful, it can be disabled on config tab.

#### Upscale

It uses "SD upscale", that means original image is split into overlapping tiles with size 512x512. Each tile is processed with SD, then they are merged into a single out image. This algorithm is very sensitive to original img resolution. For sane processing time try to use images of size up to 1408x960 (max size for 6 tiles).

You should use low denoising strength with this mode. Think 0.1-0.2.

#### Inpainting

I'm not sure, it works correctly currently. If you think it doesn't please try webui.

It requires both image and mask. For mask this plugin uses selected layer. Just create new layer and paint with white brush. This area will be inpainted.

For inpainting to work properly you need high denoising strength. Think 0.6-0.8.

#### Image resizing

In every mode except sd upscale plugin resizes source images. First image is resized to match SD required size of 512x(512 + 64*k). Second resulting image is resized back. 
That means that you should be able to use plugin with image or selection of any size. But large image sizes will generally have less downscaling artefacts.

Internally plugin uses Lanczos algorithm for both downscaling and upscaling.

#### Aspect ratio handling

**TLDR:** use selections, with them aspect ratio is less wrong.

Plugin needs to resize image to size of (512 + 64*k)x512. That can change aspect ratio and lead to suboptimal results.

If you use selection, plugin will try to slightly increase size of an image patch, which is sent to SD. This improves aspect ratio handling quite a bit.
Alternatively work with image sizes that have right aspect ratio, like 1024x1024, 1280x1024, ... like (512 + 64*k)x512.

SD upscaling doesn't have this problem at all.