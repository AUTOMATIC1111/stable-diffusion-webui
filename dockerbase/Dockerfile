FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

MAINTAINER wangdongming "wangdongming@dragonest.com"
ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive

ARG TZ=Asia/Shanghai
# 安装PYTHON
RUN apt-get update \
    && apt-get dist-upgrade -yq --no-install-recommends \
    && apt-get install -yq --no-install-recommends \
        curl \
        sudo \
        git-core \
	    git-lfs \
	    python3 \
	    python3-pip \
	    python3-venv \
	    python3-opencv
# pip install
RUN pip3 install numpy==1.23.3 psutil safetensors torchsde GitPython Pillow==9.4.0 inflection kornia torchdiffeq requests torch  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install opencv-contrib-python fonts font-roboto omegaconf gradio==3.16.2 torchdiffeq==0.2.3 lark==1.1.2 pytorch_lightning==1.7.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install transformers==4.25.1 blendmodes==2022 einops==0.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple