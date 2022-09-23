#!/usr/bin/env bash -l

# Install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install conda
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# Add conda to path
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda
conda init

# Remove previous conda environment
conda remove -n web-ui --all

# Create conda environment
conda create -n web-ui python=3.10

# Activate conda environment
conda activate web-ui

# Remove previous git repository
rm -rf stable-diffusion-webui

# Clone the repo
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# Enter the repo
cd stable-diffusion-webui

echo "============================================="
echo "============================================="
echo "===========STABLE DIFFUSION MODEL============"
echo "============================================="
echo "============================================="

# Prompt the user to ask if they've already installed the model
echo "If you've already downloaded the model, you now have time to copy it yourself to stable-diffusion-webui/models/"
echo "If you haven't downloaded the model yet, you can enter n to downloaded the model from hugging face."
while true; do
    read -p "Have you already installed the model? (y/n) " yn
    case $yn in
        [Yy]* ) echo "Skipping model installation"; break;;
        [Nn]* ) echo "Installing model"; 
        # Prompt the user for their hugging face token and store it in a variable
        echo "Register an account on huggingface.co and then create a token (read) on https://huggingface.co/settings/tokens"
        read -p "Please enter your hugging face token: " hf_token
        # Install the model
        headertoken="Authorization: Bearer $hf_token"
        curl -L -H "$headertoken" -o models/sd-v1-4.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt 
        break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# Clone required repos
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
 
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers

git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    
git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# Install dependencies
pip install -r requirements.txt

# There's a bug in protobuf that causes errors when generating images.
# Read: https://github.com/protocolbuffers/protobuf/issues/10571
# Once this gets fixed, we can remove this line
pip install protobuf==3.19.4

# Remove torch and all related packages
pip uninstall torch torchvision torchaudio -y

# Install the latest nightly build of PyTorch
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --no-deps

# Missing dependencie(s)
pip install gdown 

# Activate the MPS_FALLBACK conda environment variable
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

# We need to reactivate the conda environment for the variable to take effect
conda deactivate
conda activate web-ui

# Create a shell script to run the web ui
echo "#!/usr/bin/env bash -l

# Activate conda environment
conda activate web-ui

# Pull the latest changes from the repo
git pull --rebase

# Run the web ui
python webui.py --precision full --no-half --opt-split-attention-v1

# Deactivate conda environment
conda deactivate
" > run_webui_mac.sh

# Give run permissions to the shell script
chmod +x run_webui_mac.sh

echo "============================================="
echo "============================================="
echo "==============MORE INFORMATION==============="
echo "============================================="
echo "============================================="
echo "If you want to run the web UI again, you can run the following command:"
echo "./stable-diffusion-webui/run_webui_mac.sh"
echo "or"
echo "cd stable-diffusion-webui && ./run_webui_mac.sh"
echo "============================================="
echo "============================================="
echo "============================================="
echo "============================================="


# Run the web UI
python webui.py --precision full --no-half --opt-split-attention-v1


