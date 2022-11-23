#!/usr/bin/env bash -l

if [ -z ${NOT_FIRST_SDSETUP_RUN} ]; then
    if ! command -v conda &> /dev/null
    then
        echo "conda is not installed. Installing miniconda"

        # Install conda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

        # Install conda
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda

        # Add conda to path
        export PATH="$HOME/miniconda/bin:$PATH"

    else
        echo "conda is installed."

    fi

    # Initialize conda
    conda init

    # Rerun the shell script with a new shell (required to apply conda environment if conda init was run for the first time)
    exec bash -c "NOT_FIRST_SDSETUP_RUN=1 \"$0\""
fi

export -n NOT_FIRST_SDSETUP_RUN

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
echo "If you've already downloaded the model, you now have time to copy it yourself to stable-diffusion-webui/models/Stable-diffusion/"
echo "If you haven't downloaded the model yet, you can enter n to downloaded the model from hugging face."
while true; do
    read -p "Have you already installed the model? (y/n) " yn
    case $yn in
        [Yy]* ) echo "Skipping model installation"; break;;
        [Nn]* ) echo "Installing model"; 
        # Prompt the user for their hugging face token and store it in a variable
        echo "Register an account on huggingface.co and then create a token (read) on https://huggingface.co/settings/tokens"
        echo "Also make sure to accept the disclaimer here: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original"
        read -p "Please enter your hugging face token: " hf_token
        # Install the model
        headertoken="Authorization: Bearer $hf_token"
        curl -L -H "$headertoken" -o models/Stable-diffusion/sd-v1-4.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt 
        break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# Clone required repos
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
 
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers

git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    
git clone https://github.com/salesforce/BLIP.git repositories/BLIP

git clone https://github.com/Birch-san/k-diffusion repositories/k-diffusion

# Before we continue, check if 1) the model is in place 2) the repos are cloned
if ( [ -f "models/sd-v1-4.ckpt" ] || [ -f "models/Stable-diffusion/sd-v1-4.ckpt" ] ) && [ -d "repositories/stable-diffusion" ] && [ -d "repositories/taming-transformers" ] && [ -d "repositories/CodeFormer" ] && [ -d "repositories/BLIP" ]; then
    echo "All files are in place. Continuing installation."
else
    echo "============================================="
    echo "====================ERROR===================="
    echo "============================================="
    echo "The check for the models & required repositories has failed."
    echo "Please check if the model is in place and the repos are cloned."
    echo "You can find the model in stable-diffusion-webui/models/Stable-diffusion/sd-v1-4.ckpt"
    echo "You can find the repos in stable-diffusion-webui/repositories/"
    echo "============================================="
    echo "====================ERROR===================="
    echo "============================================="
    exit 1
fi

# Install dependencies
pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1

pip install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379

pip install torch==1.12.1 torchvision==0.13.1

pip install torchsde

# Patch the bug that prevents torch from working (see https://github.com/Birch-san/stable-diffusion#patch), rather than try to use a nightly build
echo "--- a/functional.py	2022-10-14 05:28:39.000000000 -0400
+++ b/functional.py	2022-10-14 05:39:25.000000000 -0400
@@ -2500,7 +2500,7 @@
         return handle_torch_function(
             layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
         )
-    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
+    return torch.layer_norm(input.contiguous(), normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
 
 
 def group_norm(
" | patch -p1 -d "$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")"/nn

# Missing dependencie(s)
pip install gdown fastapi psutil

# Activate the MPS_FALLBACK conda environment variable
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

# We need to reactivate the conda environment for the variable to take effect
conda deactivate
conda activate web-ui

# Check if the config var is set
if [ -z "$PYTORCH_ENABLE_MPS_FALLBACK" ]; then
    echo "============================================="
    echo "====================ERROR===================="
    echo "============================================="
    echo "The PYTORCH_ENABLE_MPS_FALLBACK variable is not set."
    echo "This means that the script will either fall back to CPU or fail."
    echo "To fix this, please run the following command:"
    echo "conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1"
    echo "Or, try running the script again."
    echo "============================================="
    echo "====================ERROR===================="
    echo "============================================="
    exit 1
fi

# Create a shell script to run the web ui
echo "#!/usr/bin/env bash -l

# This should not be needed since it's configured during installation, but might as well have it here.
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate conda environment
conda activate web-ui

# Pull the latest changes from the repo
git pull --rebase

# Run the web ui
python webui.py --precision full --no-half --use-cpu Interrogate GFPGAN CodeFormer \$@

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
python webui.py --precision full --no-half --use-cpu Interrogate GFPGAN CodeFormer


