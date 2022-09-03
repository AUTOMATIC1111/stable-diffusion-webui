# Stable Diffusion Krita Plugin
A simple interface based on this repository: https://github.com/AUTOMATIC1111/stable-diffusion-webui

Requires Krita 5.1

https://user-images.githubusercontent.com/112324253/188290881-0ee2cbf2-e1a0-422e-9c2c-4b99b3a5f723.mp4

## Installing and running

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
- place `model.ckpt` into webui directory, next to `webui.bat`.
- _*(optional)*_ place `GFPGANv1.3.pth` into webui directory, next to `webui.bat`.
- run `webui.bat` from Windows Explorer.

#### Troublehooting:

Look into parent repository https://github.com/AUTOMATIC1111/stable-diffusion-webui for instructions. You should make sure webui.cmd works.

### Usage

Put something in your prompt and just run it. If you select some area, only it will be used.

#### Hotkeys

- txt2img - Ctrl + Alt + Q
- img2img - Ctrl + Alt + W
- sd upscale - Ctrl + Alt + E
- inpainting - Ctrl + Alt + R

#### Img2img

You may use feathered selections. Selection will be converted to selection mask afterwards. Not sure it is useful, it can be disabled on config tab.

#### Upscale

It uses "SD upscale", that means original image is split into overlapping tiles with size 512x512. Each tile is processed with SD, then they are merged into a single out image. This algorithm is very sensitive to original img resolution. For sane processing time try to use images of size up to 1980x1080.

#### Inpainting

It requires both image and mask. For mask this plugin uses selected layer. Just create new layer and paint with white brush. This are will be inpainted.
