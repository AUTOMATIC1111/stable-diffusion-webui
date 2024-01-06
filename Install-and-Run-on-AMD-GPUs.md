# Windows
Windows+AMD support has **not** officially been made for webui, \
but you can install lshqqytiger's fork of webui that uses **Direct-ml**. 

Training currently doesn't work, yet a variety of features/extensions do, such as LoRAs and controlnet. Report issues at https://github.com/lshqqytiger/stable-diffusion-webui-directml/issues 

1. Install [Python 3.10.6](https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe) (ticking **Add to PATH**), and [git](https://github.com/git-for-windows/git/releases/download/v2.39.2.windows.1/Git-2.39.2-64-bit.exe)
2. paste this line in cmd/terminal: `git clone https://github.com/lshqqytiger/stable-diffusion-webui-directml && cd stable-diffusion-webui-directml && git submodule init && git submodule update` \
<sup>(you can move the program folder somewhere else.)</sup> 
3. Double-click webui-user.bat
4. If it looks like it is stuck when installing or running, press enter in the terminal and it should continue.

If you have 4-6gb vram, try adding these flags to `webui-user.bat` like so:

`COMMANDLINE_ARGS=--opt-sub-quad-attention --lowvram --disable-nan-check`

(The rest **below are installation guides for linux** with rocm.)


# Automatic Installation

<sup>(As of [1/15/23](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6709) you can just run `webui.sh` and pytorch+rocm should be automatically installed for you.)</sup>


1. Enter these commands, which will install webui to your current directory:

```
sudo apt install git python3.10-venv -y
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui && cd stable-diffusion-webui
python3.10 -m venv venv
```

2. Install and run with:

    ./webui.sh {your_arguments*}

*For many AMD GPUs, you **must** add `--precision full --no-half` or `--upcast-sampling` arguments to avoid NaN errors or crashing. 
If `--upcast-sampling` works as a fix with your card, you should have 2x speed (fp16) compared to running in full precision.

<details>

- Some cards like the Radeon RX 6000 Series and the RX 500 Series will already run fp16 perfectly fine (noted [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/5468).)

- If your card is unable to run SD with the latest pytorch+rocm core package, you can try installing previous versions, by following a more manual installation guide below.

</details>


# Running natively

Execute the following:

```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' python launch.py --precision full --no-half
```

In following runs you will only need to execute:
```bash
cd stable-diffusion-webui
# Optional: "git pull" to update the repository
source venv/bin/activate

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' python launch.py --precision full --no-half
```

The first generation after starting the WebUI might take very long, and you might see a message similar to this: 
> MIOpen(HIP): Warning [SQLiteBase] Missing system database file: gfx1030_40.kdb Performance may degrade. Please follow
> instructions to install: https://github.com/ROCmSoftwarePlatform/MIOpen#installing-miopen-kernels-package

The next generations should work with regular performance. You can follow the link in the message, and if you happen
to use the same operating system, follow the steps there to fix this issue. If there is no clear way to compile or
install the MIOpen kernels for your operating system, consider following the "Running inside Docker"-guide below.



# Running inside Docker
Pull the latest `rocm/pytorch` Docker image, start the image and attach to the container (taken from the `rocm/pytorch`
documentation): `docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/pytorch`

Execute the following inside the container:
```bash
cd /dockerx
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
python -m pip install --upgrade pip wheel

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

Following runs will only require you to restart the container, attach to it again and execute the following inside the
container: Find the container name from this listing: `docker container ls --all`, select the one matching the
`rocm/pytorch` image, restart it: `docker container restart <container-id>` then attach to it: `docker exec -it
<container-id> bash`.

```bash
cd /dockerx/stable-diffusion-webui
# Optional: "git pull" to update the repository

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

The `/dockerx` folder inside the container should be accessible in your home directory under the same name.

## Updating Python version inside Docker
If the web UI becomes incompatible with the pre-installed Python 3.7 version inside the Docker image, here are
instructions on how to update it (assuming you have successfully followed "Running inside Docker"):

Execute the following inside the container:
```bash
apt install python3.9-full # Confirm every prompt
update-alternatives --install /usr/local/bin/python python /usr/bin/python3.9 1
echo 'PATH=/usr/local/bin:$PATH' >> ~/.bashrc
```

Run `source ~/.bashrc` and proceed by running the same commands as you would with an existing container.

It's possible that you don't need "--precision full", dropping "--no-half" however it may not work for everyone. Certain cards like the Radeon RX 6000 Series and the RX 500 Series will function normally without the option `--precision full --no-half`, saving plenty of VRAM. (noted [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/5468).)

# Install on AMD and Arch Linux

**Install webui on Arch Linux with Arch-specific packages**  
*and possibly other Arch-based Linux distributions (tested Feb 22 2023)*

## Arch-specific dependencies

1. Start with [required dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies#required-dependencies) and install `pip`

```bash
sudo pacman -S python-pip
```

2. Install `pytorch` with ROCm backend

Arch [Community] repository offers two `pytorch` packages, `python-pytorch-rocm` and `python-pytorch-opt-rocm`. For CPUs with AVX2 instruction set support, that is, CPU microarchitectures beyond Haswell (Intel, 2013) or Excavator (AMD, 2015), install `python-pytorch-opt-rocm` to benefit from performance optimizations. Otherwise install `python-pytorch-rocm`:

```bash
# Install either one:
sudo pacman -S python-pytorch-rocm
sudo pacman -S python-pytorch-opt-rocm   # AVX2 CPUs only
```

3. Install `torchvision` with ROCm backend

`python-torchvision-rocm` package is located in AUR. Clone the git repository and compile the package on your machine

```bash
git clone https://aur.archlinux.org/python-torchvision-rocm.git
cd python-torchvision-rocm
makepkg -si
```

Confirm all steps until Pacman finishes installing `python-torchvision-rocm`.

Alternatively, install the `python-torchvision-rocm` package with a [AUR helper](https://wiki.archlinux.org/title/AUR_helpers).

## Setup `venv` environment

1. Manually create a `venv` environment with system site-packages (this will allows access to system `pytorch` and `torchvision`). Install the remaining Python dependencies

```bash
python -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

## Launch

Run the following inside the project root to start webui:

```bash
source venv/bin/activate
./webui.sh
```

Depending on the GPU model, you may need to add certain [Command Line Arguments](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings) and [Optimizations](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations) to `webui-user.sh` in order for webui to run properly. Refer to the [Automatic Installation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#automatic-installation) section.

## Limitations

-  GPU model has to be supported by Arch dependencies

See if your GPU is listed as a build architecture in `PYTORCH_ROCM_ARCH` variable for [Tourchvision](https://github.com/rocm-arch/python-torchvision-rocm/blob/b66f7ed9540a0e25f4a81bf0d9cfc3d76bc0270e/PKGBUILD#L68-L74) and [PyTorch](https://github.com/archlinux/svntogit-community/blob/5689e7f44f082ba3c37724c2890e93e7106002a1/trunk/PKGBUILD#L220). References for architectures can be found [here](https://llvm.org/docs/AMDGPUUsage.html#processors). If not, consider building both packages locally or use another [installation method](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs).

- Arch dependencies (`pytorch`, `torchvision`) are kept up-to-date by full system updates (`pacman -Syu`) and compiling, which may not be desirable when dependency combinations with fixed versions are wished

*This guide has been tested on AMD Radeon RX6800 with Python 3.10.9, ROCm 5.4.3, PyTorch 1.13.1, Torchvision 0.14.1*
