Before attempting to install make sure all the required [dependencies](Dependencies) are met.

# Automatic Installation
## Windows
Run `webui-user.bat` from Windows Explorer as normal, ***non-administrator***, user.

See [Troubleshooting](Troubleshooting) section for what to do if things go wrong.

## Linux
To install in the default directory `/home/$(whoami)/stable-diffusion-webui/`, run:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```

In order to customize the installation, clone the repository into the desired location, change the required variables in `webui-user.sh` and run :
```bash
bash webui.sh
```

## Third party installation guides/scripts:
- NixOS: https://github.com/virchau13/automatic1111-webui-nix

# Almost Automatic Installation and Launch
To install the required packages via pip without creating a virtual environment, run:
```bash
python launch.py
```

Command line arguments may be passed directly, for example:
```bash
python launch.py --opt-split-attention --ckpt ../secret/anime9999.ckpt
```

# Manual Installation
Manual installation is very outdated and probably won't work. check colab in the repo's readme for instructions.

The following process installs everything manually on both Windows or Linux (the latter requiring `dir` to be replaced by `ls`):
```bash
# install torch with CUDA support. See https://pytorch.org/get-started/locally/ for more instructions if this fails.
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

# check if torch supports GPU; this must output "True". You need CUDA 11. installed for this. You might be able to use
# a different version, but this is what I tested.
python -c "import torch; print(torch.cuda.is_available())"

# clone web ui and go into its directory
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# clone repositories for Stable Diffusion and (optionally) CodeFormer
mkdir repositories
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# install requirements of Stable Diffusion
pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary

# install k-diffusion
pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary

# (optional) install GFPGAN (face restoration)
pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary

# (optional) install requirements for CodeFormer (face restoration)
pip install -r repositories/CodeFormer/requirements.txt --prefer-binary

# install requirements of web ui
pip install -r requirements.txt  --prefer-binary

# update numpy to latest version
pip install -U numpy  --prefer-binary

# (outside of command line) put stable diffusion model into web ui directory
# the command below must output something like: 1 File(s) 4,265,380,512 bytes
dir model.ckpt

```

The installation is finished, to start the web ui, run:
```bash
python webui.py
```

# Windows 11 WSL2 instructions
To install under a Linux distro in Windows 11's WSL2:
```bash
# install conda (if not already done)
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh

# Clone webui repo
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Create and activate conda env
conda env create -f environment-wsl2.yaml
conda activate automatic

```

At this point, the instructions for the Manual installation may be applied starting at step `# clone repositories for Stable Diffusion and (optionally) CodeFormer`.


# Alternative installation on Windows using Conda
- Prerequisites _*(Only needed if you do not have them)*_. Assumes [Chocolatey](https://chocolatey.org/install) is installed. 
    ```bash
    # install git
    choco install git
    # install conda
    choco install anaconda3
    ```
    Optional parameters: [git](https://community.chocolatey.org/packages/git), [conda](https://community.chocolatey.org/packages/anaconda3)
- Install (warning: some files exceed multiple gigabytes, make sure you have space first)
  1. Download as .zip and extract or use git to clone.
        ```bash
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
        ```
  2. Launch the Anaconda prompt. It should be noted that you can use older Python versions, but you may be forced to manually remove features like cache optimization, which will degrade your performance.
        ```bash
        # Navigate to the git directory
        cd "GIT\StableDiffusion"
        # Create environment
        conda create -n StableDiffusion python=3.10.6
        # Activate environment
        conda activate StableDiffusion
        # Validate environment is selected
        conda env list
        # Start local webserver
        webui-user.bat
        # Wait for "Running on local URL:  http://127.0.0.1:7860" and open that URI.
        ```
    3. _*(Optional)*_ Go to [CompVis](https://huggingface.co/CompVis) and download latest model, for example [1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) and unpack it to ex:
        ```bash
        GIT\StableDiffusion\models\Stable-diffusion
        ```
        after that restart the server by restarting Anaconda prompt and 
        ```bash
        webui-user.bat
        ```
- Alternative defaults worth trying out:
    1. Try **euler a** (Ancestral Euler) with higher **Sampling Steps** ex: 40 or others with 100. 
    2. Set "Settings > User interface > Show image creation progress every N sampling steps" to 1 and pick a deterministic **Seed** value. Can visually see how image defusion happens and record a .gif with [ScreenToGif](https://github.com/NickeManarin/ScreenToGif).
    3. Use **Restore faces**. Generally, better results, but that quality comes at the cost of speed.

