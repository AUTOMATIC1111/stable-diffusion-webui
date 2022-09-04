# Stable Diffusion Krita Plugin
A simple interface based on this repository: https://github.com/AUTOMATIC1111/stable-diffusion-webui

Requires Krita 5.1

## Usage example
[target.webm](https://user-images.githubusercontent.com/112324253/188291339-9d146a9a-ba9f-4671-9bd8-c8b55fd48ba6.webm)


## Installing and running

If you used previous version which used conda, please make a new install. Please install it separately from webui.

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
installed to run this, and an NVidia videocard.

I tested the installation to work Windows with Python 3.8.10, and with Python 3.10.6. You may be able
to have success with different versions.

You need `model.ckpt`, Stable Diffusion model checkpoint, a big file containing the neural network weights. You
can obtain it from the following places:
 - [official download](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
 - [file storage](https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl)
 - magnet:?xt=urn:btih:3a4a612d75ed088ea542acac52f9f45987488d1c&dn=sd-v1-4.ckpt&tr=udp%3a%2f%2ftracker.openbittorrent.com%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337

You optionally can use GPFGAN to improve faces, then you'll need to download the model from [here](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth).

### Automatic installation/launch

- install [Python 3.10.6](https://www.python.org/downloads/windows/)
- install [git](https://git-scm.com/download/win)
- install [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64)
- place `model.ckpt` into webui directory, next to `krita.bat`.
- _*(optional)*_ place `GFPGANv1.3.pth` into webui directory, next to `krita.bat`.
- run `krita.bat` from Windows Explorer.

#### Troublehooting:

Look into parent repository https://github.com/AUTOMATIC1111/stable-diffusion-webui for instructions. You should make sure webui.cmd works.

### Usage

Put something in your prompt and just run it. If you select some area, only it will be used.

#### Hotkeys

- txt2img - Ctrl + Alt + Q
- img2img - Ctrl + Alt + W
- sd upscale - Ctrl + Alt + E
- inpainting - Ctrl + Alt + R

#### Aspect ratio handling

Plugin needs to resize image to size of (512 + 64*k)x512. That can change aspect ratio and lead to suboptimal results.

If you use selection, plugin will try to slightly increase size of an image patch, which is sent to SD. This improves aspect ratio handling quite a bit. 
Alternatively work with image sizes that have right aspect ratio, like 1024x1024, 1280x1024, ... like (512 + 64*k)x512.

TLDR; use selections, with them aspect ratio is less wrong.

SD upscaling doesn't have this problem at all.

#### Img2img

You may use feathered selections. Selection will be converted to transparency mask afterwards. Not sure it is useful, it can be disabled on config tab.

#### Upscale

It uses "SD upscale", that means original image is split into overlapping tiles with size 512x512. Each tile is processed with SD, then they are merged into a single out image. This algorithm is very sensitive to original img resolution. For sane processing time try to use images of size up to 1980x1080.

You should use low denoising strength with this mode. Think 0.1-0.2.

#### Inpainting

It requires both image and mask. For mask this plugin uses selected layer. Just create new layer and paint with white brush. This area will be inpainted.

For inpainting to work properly you need high denoising strength. Think 0.6-0.8.
