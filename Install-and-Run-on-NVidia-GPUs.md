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

# (outside of command line) put the GFPGAN model into web ui directory
# the command below must output something like: 1 File(s) 348,632,874 bytes
dir GFPGANv1.3.pth
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

# (optional) install requirements for GFPGAN (upscaling)
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
```

At this point, the instructions for the Manual installation may be applied starting at step `# clone repositories for Stable Diffusion and (optionally) CodeFormer`.
