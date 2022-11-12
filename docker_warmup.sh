#!/bin/bash

echo "Warmup: installing SD dependencies and downloading models"

if [[ -z "${HUGGINGFACE_HUB_TOKEN}" ]]; then
    echo "HUGGINGFACE_HUB_TOKEN environment variable missing, can't download models"
    exit 1
fi

# Create venv if not exists, then activate it
python -m venv venv
source venv/bin/activate

# Install xformers
pip install /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl --no-deps

# Install dependencies
python launch.py --exit --skip-torch-cuda-test

# Replace opencv-python (installed as a side effect of `python launch.py) with
# opencv-python-headless, to remove dependency on missing libGL.so.1.
pip install opencv-python-headless

# Install dependencies
python -c "import torch; torch.save({}, 'model.ckpt')"
python -c "import webui; webui.initialize()"
rm /sd/model.ckpt

# Download CodeFormer models
python -c "import webui; \
    webui.codeformer.setup_model(webui.cmd_opts.codeformer_models_path); \
    webui.shared.face_restorers[0].create_models();"

# Download GFPGAN models
python -c "import webui; \
    webui.gfpgan.setup_model(webui.cmd_opts.gfpgan_models_path); \
    webui.gfpgan.gfpgann()"

# Download ESRGAN models
python -c "import webui; \
    from modules.esrgan_model import UpscalerESRGAN; \
    upscaler = UpscalerESRGAN('/sd/models/ESRGAN'); \
    upscaler.load_model(upscaler.model_url)"

# Array of model files to pre-download
# local filename
# download URL
# Inspired by https://github.com/sd-webui/stable-diffusion-webui/blob/master/entrypoint.sh
SD_MODEL_FILES=(
    'sd.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt'
    'sd.vae.pt https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt'
    'waifu.ckpt https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt'
    'waifu.vae.pt https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt'
    'yiffy.ckpt https://sexy.canine.wf/file/yiffy-ckpt/yiffy-e18.ckpt'
    'gg1342.ckpt https://files.stablediffusionhub.com/model/gg1342_testrun1_pruned.ckpt'
    'f111.ckpt https://storage.googleapis.com/digburn/f111.ckpt'
)

echo "Downloading model files..."
for models in "${SD_MODEL_FILES[@]}"; do
    model=($models)
    file=${model[0]}
    path="/sd/models/Stable-diffusion/${file}"
    url=${model[1]}

    if [[ $url =~ "huggingface.co" ]]; then
        wget --header "Authorization: Bearer $HUGGINGFACE_HUB_TOKEN" -O ${path} ${url}
    else
        wget -O ${path} ${url}
    fi

    if [[ -e "${path}" ]]; then
        echo "Saved ${file}"
    else
        echo "Error saving ${path}!"
        exit 1
    fi
done

# Use SD 1.5 VAE on photorealistic models
cd /sd/models/Stable-diffusion
cp sd.vae.pt gg1342.vae.pt
cp sd.vae.pt f111.vae.pt

cd /sd

echo "Warmup: done"