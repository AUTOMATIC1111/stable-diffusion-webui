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


FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

# Store MatPlotLib config files in the app directory
ENV MPLCONFIGDIR=/sd/.mpl_config
# Store Transformers models in the app directory
ENV XDG_CACHE_HOME=/sd/.xdg_cache

RUN apt update && apt install git wget -y

COPY --from=xformers xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl

COPY --chown=1000:1000 . /sd
WORKDIR /sd

RUN mkdir /var/sd
RUN chown -R 1000 /var/sd

USER 1000:1000

# Python dependencies are installed by the entrypoint to keep the docker image as small as possible.
# To make startup faster, it's possible to mount a warmed-up SD directory at /var/sd.
CMD ["bash", "docker_entrypoint.sh"]
