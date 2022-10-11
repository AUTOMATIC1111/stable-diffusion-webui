# Update ubuntu
sudo apt update
sudo apt upgrade
sudo apt install git wget

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x ./Anaconda3-2022.05-Linux-x86_64.sh
# Accept license and install

yes | ./Anaconda3-2022.05-Linux-x86_64.sh

# Nvidia CUDA toolkit for Ubuntu WSL - https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-wsl-ubuntu-11-3-local_11.3.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-3-local_11.3.1-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Install g++
sudo apt install build-essential deepspeed

# New virtual Python environment
conda update -n base -c defaults conda
conda create --name diffusers python=3.9
conda activate diffusers

# Make a directory for all your github downloads, then download diffusers
mkdir ~/github
cd ~/github
git clone https://github.com/Ttl/diffusers.git
cd diffusers
git checkout dreambooth_deepspeed
git pull

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -U --pre triton
pip install ninja bitsandbytes
pip install git+https://github.com/facebookresearch/xformers@1d31a3a#egg=xformers
pip install deepspeed
pip install diffusers

accelerate config

# Use following options

#In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
#Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 0
#Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:
#Do you want to use DeepSpeed? [yes/NO]: yes
#Do you want to specify a json file to a DeepSpeed config? [yes/NO]:
#What should be your DeepSpeed's ZeRO optimization stage (0, 1, 2, 3)? [2]:
#Where to offload optimizer states? [none/cpu/nvme]: cpu
#Where to offload parameters? [none/cpu/nvme]: cpu
#How many gradient accumulation steps you're passing in your script? [1]:
#Do you want to use gradient clipping? [yes/NO]:
#Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
#How many GPU(s) should be used for distributed training? [1]:
#Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: fp16

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
