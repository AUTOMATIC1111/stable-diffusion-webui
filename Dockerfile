# Build container
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

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Build and install xformers
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" \
    pip3 wheel git+https://github.com/facebookresearch/xformers.git#egg=xformers

# Run container
FROM python:3.10

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

COPY . /sd

WORKDIR /sd

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
RUN python launch.py --skip-torch-cuda-test --exit

# Replace opencv-python (installed as a side effect of `python launch.py`) with opencv-python-headless,
# to remove dependency on missing libGL.so.1.
RUN pip install opencv-python-headless

# Install prebuilt xformers
COPY --from=xformers xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl
RUN pip install /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl

# Install nvidia build tool nvcc and required libraries to build xformers
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204-keyring.gpg \
#    -O /usr/share/keyrings/cuda-archive-keyring.gpg && \
#    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | \
#    tee /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list && \
#    apt-get update && \
#    apt-get install -y cuda-nvcc-11-8 libcusparse-dev-11-8 libcublas-dev-11-8 libcusolver-dev-11-8 libcurand-dev-11-8
#
## Build and install xformers
#RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" \
#    pip install git+https://github.com/facebookresearch/xformers.git#egg=xformers

# Download supporting models (e.g. the very large openai/clip-vit-large-patch14)
# Create a dummy model to pass the "sd model exists" check, so SD continues initialization
RUN python -c "import torch; torch.save({}, 'model.ckpt')" \
    && python -c "import webui; webui.initialize()" \
    && rm /sd/model.ckpt

CMD ["python", "launch.py", "--listen", "--xformers"]
