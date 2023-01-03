# Required Dependencies
1. Python 3.10.6 and Git:
    - Windows: download and run installers for Python 3.10.6 ([webpage](https://www.python.org/downloads/release/python-3106/), [exe](https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe), or [win7 version](https://github.com/adang1345/PythonWin7/raw/master/3.10.6/python-3.10.6-amd64-full.exe)) and git ([webpage](https://git-scm.com/download/win))
    - Linux (Debian-based): `sudo apt install wget git python3 python3-venv`
    - Linux (Red Hat-based): `sudo dnf install wget git python3`
    - Linux (Arch-based): `sudo pacman -S wget git python3`
2. Code from this repository:
    - preferred way: using git: `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
        - This way is referred because it lets you update by just running `git pull`.
        - Those commands can be used from command line window that opens after you right click in Explorer and select "Git Bash here".
    - alternative way: use the "Code" (green button) -> "Download ZIP" option on the main page of the repo.
        - You still need to install git even if you choose this.
        - To update, you'll have to download zip again and replace files.
3. The Stable Diffusion model checkpoint, a file with `.ckpt` extension, needs to be downloaded and placed in the `models/Stable-diffusion` directory.
    - [Official download](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
    - [File storage](https://drive.yerf.org/wl/?id=EBfTrmcCCUAGaQBXVIj5lJmEhjoP1tgl)
    - Torrent (magnet:?xt=urn:btih:3a4a612d75ed088ea542acac52f9f45987488d1c&dn=sd-v1-4.ckpt&tr=udp%3a%2f%2ftracker.openbittorrent.com%3a6969%2fannounce&tr=udp%3a%2f%2ftracker.opentrackr.org%3a1337)

# Optional Dependencies

## ESRGAN (Upscaling)
ESRGAN models such as those from the [Model Database](https://upscale.wiki/wiki/Model_Database), may be placed into the ESRGAN directory. ESRGAN directory doesn't exist in the repo, until you run for the first time.

The models will be loaded as a model if it has `.pth` extension, and it will show up with its name in the UI.

> Note: RealESRGAN models are not ESRGAN models, they are not compatible. Do not download RealESRGAN models. Do not place RealESRGAN into the directory with ESRGAN models.

## .yaml files for [sd 2.x models](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20)

- 768-v-ema.ckpt [config](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml))

- 512-base-ema.ckpt [config](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml)

- 512-depth-ema.ckpt [config](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-midas-inference.yaml)

Download config .yaml file and store it in same folder as .ckpt with the same name.