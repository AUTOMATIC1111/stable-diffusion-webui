# xformers build container
FROM nvidia/cuda:11.3.1-devel-ubuntu18.04 as xformers

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y curl git python3.10 python3.10-dev

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install pip using the standalone installer, as the apt package installs python 3.6
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install xformers dependencies
RUN pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Build and install xformers
RUN git clone --recursive https://github.com/facebookresearch/xformers.git
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" \
    pip3 wheel --no-deps -e xformers


# base container, tracks main branch
FROM python:3.10 AS base

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install prebuilt xformers
COPY --from=xformers xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl
RUN pip install /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl

# Clone main branch
RUN git clone https://github.com/Equilibrium-Point/automatic1111-stable-diffusion.git /sd

WORKDIR /sd

# Install dependencies
RUN python launch.py --skip-torch-cuda-test --exit

# Replace opencv-python (installed as a side effect of `python launch.py`) with opencv-python-headless,
# to remove dependency on missing libGL.so.1.
RUN pip install opencv-python-headless


# download container, tracks main branch
FROM base AS download

# Download supporting models (e.g. the very large openai/clip-vit-large-patch14)
# Create a dummy model to pass the "sd model exists" check, so SD continues initialization
RUN python -c "import torch; torch.save({}, 'model.ckpt')" \
    && python -c "import webui; webui.initialize()" \
    && rm /sd/model.ckpt

# Download CodeFormer models
RUN python -c "import webui; \
    webui.codeformer.setup_model(webui.cmd_opts.codeformer_models_path); \
    webui.shared.face_restorers[0].create_models();"

# Download GFPGAN models
RUN python -c "import webui; \
    webui.gfpgan.setup_model(webui.cmd_opts.gfpgan_models_path); \
    webui.gfpgan.gfpgann()"

# Download ESRGAN models
RUN python -c "import webui; \
    from modules.esrgan_model import UpscalerESRGAN; \
    upscaler = UpscalerESRGAN('/sd/models/ESRGAN'); \
    upscaler.load_model(upscaler.model_url)"


# Slim container, applies local code, no downloads
FROM base AS slim

COPY . /sd

CMD ["python", "launch.py", "--api", "--listen", "--xformers"]


# Run container, applies local code, with downloads
FROM download

COPY . /sd

CMD ["python", "launch.py", "--api", "--listen", "--xformers"]