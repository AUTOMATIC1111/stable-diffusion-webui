# Required Dependencies
1. Python 3.10.6 and Git:
    - Windows:
        - [Python](https://www.python.org/downloads/windows/)
        - [Git](https://git-scm.com)
    - Linux (Debian-based):
    ```bash
    sudo apt install wget git python3 python3-venv
    ```
    - Linux (Red Hat-based):
    ```bash
    sudo dnf install wget git python3
    ```
    - Linux (Arch-based):
    ```bash
    sudo pacman -S wget git python3
    ```
2. The stable-diffusion-webui code may be cloned by running:
    ```bash
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    ```
3. The Stable Diffusion model checkpoint `model.ckpt` needs to be placed in the base directory, alongside `webui.py`
    - [Official download](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
    - [File storage](https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl)
    - Torrent (magnet:?xt=urn:btih:3a4a612d75ed088ea542acac52f9f45987488d1c&dn=sd-v1-4.ckpt&tr=udp%3a%2f%2ftracker.openbittorrent.com%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337)

# Optional Dependencies
## GFPGAN (Improve Faces)
GFPGAN can be used to improve faces, requiring the [model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) to be placed in the base directory.

## ESRGAN (Upscaling)
ESRGAN models such as those from the [Model Database](https://upscale.wiki/wiki/Model_Database), may be placed into the ESRGAN directory.
A file will be loaded as a model if it has `.pth` extension, and it will show up with its name in the UI.

> Note: RealESRGAN models are not ESRGAN models, they are not compatible. Do not download RealESRGAN models. Do not place RealESRGAN into the directory with ESRGAN models.
