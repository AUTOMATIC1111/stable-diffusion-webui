# sd-web ui base environment image.
# build command:
# docker build -t sd-webui-env:v0.1 . --build-arg HTTP_PROXY=http://172.16.255.22:3128

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
	    python3-opencv  \
        wget

RUN apt-get install tk potrace ffmpeg unzip gnupg2 -y
# pip install
RUN pip3 install numpy==1.23.3 psutil safetensors==0.2.7 torchsde==0.2.5 GitPython==3.1.27 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install Pillow==9.4.0 inflection==0.5.1 kornia==0.6.7 torchdiffeq requests torch==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install fastapi==0.90.1 gradio==3.16.2 torchdiffeq==0.2.3 lark==1.1.2  xformers==0.0.16rc425 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install transformers==4.25.1 blendmodes==2022 einops==0.4.1 markupsafe open-clip-torch==1.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install opencv-contrib-python fonts font-roboto omegaconf  pytorch_lightning==1.7.6  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install omegaconf==2.2.3 fairscale==0.4.4 timm==0.4.12 pudb==2019.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install pycocoevalcap scipy -i https://mirrors.aliyun.com/pypi/simple
RUN pip3 install tqdm wandb scipy clip-anytorch accelerate facexlib open_clip_torch -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install ftfy regex -i https://mirrors.aliyun.com/pypi/simple
# 以下方式2选一
#RUN pip3 install open_clip_torch gfpgan  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 方法2：
ARG HTTP_PROXY
RUN https_proxy=${HTTP_PROXY} python3 -m pip install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 --prefer-binary
# RUN https_proxy=${HTTP_PROXY} python3 -m pip install git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b --prefer-binary
RUN https_proxy=${HTTP_PROXY} python3 -m pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 --prefer-binary
CMD ["bash"]