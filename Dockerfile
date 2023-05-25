# build cmd: docker build -t sd-webui . --build-arg HTTP_PROXY=http://172.16.255.22:3128
# FROM i.harbor.dragonest.net/xingzhe/sd-webui

# build dockerbase/Dockerfile first
# run cmd:
# nvidia-docker run -p 7860:7860 -it -v /data/apksamba/sd/models:/root/stable-diffusion-webui/models i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui:v0.7 /bin/bash
# nvidia-docker run -d -p 7860:7860 -v /data/apksamba/sd/models:/root/stable-diffusion-webui/models --env-file /data/apksamba/sd/env.dev i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui:v0.10
FROM i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui-env:v0.1

MAINTAINER wangdongming "wangdongming@dragonest.com"

ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
ENV TORCH_HOME /root/stable-diffusion-webui/models/torch

ARG TZ=Asia/Shanghai
ARG BUILD_ARGS
ARG HTTP_PROXY

ARG VOLUME
ENV BUILD_ARGS ${BUILD_ARGS}
ENV VOLUME data

ARG USERNAME=wow
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 安装PYTHON
RUN apt-get update \
    && apt-get dist-upgrade -yq --no-install-recommends \
    && apt-get install -yq --no-install-recommends \
        curl \
        sudo \
        git-core \
	    git-lfs \
	    libgl1 \
	    libglib2.0-0 \
        python3-opencv\
        python3.10\
        python3-pip\
        python3-venv \
        wget

RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple/

# 从GITHUB下载代码，在GITHUB托管方便合并最新代码
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Jackstrawcd/stable-diffusion-webui.git ~/stable-diffusion-webui

# 安装requirements
RUN cd /root/stable-diffusion-webui  \
    && pip3 install xformers -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple \
    && pip3 install -r requirements.txt -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple \
    && pip3 install -r requirements_versions.txt -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple
# repositories
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/CompVis/taming-transformers.git /root/stable-diffusion-webui/repositories/taming-transformers
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Stability-AI/stablediffusion.git /root/stable-diffusion-webui/repositories/stable-diffusion-stability-ai
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/crowsonkb/k-diffusion.git /root/stable-diffusion-webui/repositories/k-diffusion
RUN https_proxy=${HTTP_PROXY} git clone -b v0.1.0 https://github.com/sczhou/CodeFormer.git /root/stable-diffusion-webui/repositories/CodeFormer
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/salesforce/BLIP.git /root/stable-diffusion-webui/repositories/BLIP

RUN pip3 install setuptools_rust -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r  /root/stable-diffusion-webui/repositories/CodeFormer/requirements.txt  -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple
RUN pip3 install -r  /root/stable-diffusion-webui/repositories/k-diffusion/requirements.txt  -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple

RUN cd  /root/stable-diffusion-webui/repositories/stable-diffusion-stability-ai \
    && python3 setup.py install
#RUN /root/stable-diffusion-webui/repositories/taming-transformers  \
#    && python3 setup.py install
#RUN cd  ~/stable-diffusion-webui/repositories/BLIP/ && \
#    python3 setup.py install

# ControlNet && extensions

RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Mikubill/sd-webui-controlnet.git /root/stable-diffusion-webui/extensions/sd-webui-controlnet
# builtin
# RUN https_proxy=${HTTP_PROXY} git clone https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 /root/stable-diffusion-webui/extensions/multidiffusion-upscaler
RUN https_proxy=${HTTP_PROXY} git clone https://huggingface.co/webui/ControlNet-modules-safetensors /root/stable-diffusion-webui/models/ControlNet
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN /root/stable-diffusion-webui/extensions/stable-diffusion-webui-localization-zh_CN
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git /root/stable-diffusion-webui/extensions/tagger
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/yfszzx/stable-diffusion-webui-images-browser /root/stable-diffusion-webui/extensions/images-browser
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/fkunn1326/openpose-editor.git /root/stable-diffusion-webui/extensions/openpose-editor
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/hnmr293/posex /root/stable-diffusion-webui/extensions/posex
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/hnmr293/sd-webui-llul /root/stable-diffusion-webui/extensions/sd-webui-llul
RUN https_proxy=${HTTP_PROXY} git clone https://jihulab.com/hunter0725/sd-webui-additional-networks /root/stable-diffusion-webui/extensions/additional-networks
RUN https_proxy=${HTTP_PROXY} git clone https://jihulab.com/hunter0725/stable-diffusion-webui-tokenizer /root/stable-diffusion-webui/extensions/stable-diffusion-webui-tokenizer
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg.git /root/stable-diffusion-webui/extensions/rembg
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen /root/stable-diffusion-webui/extensions/stable-diffusion-webui-promptgen
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete /root/stable-diffusion-webui/extensions/tagcomplete
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/KutsuyaYuki/ABG_extension /root/stable-diffusion-webui/extensions/ABG_extension

RUN pip3 install segment_anything -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple
RUN wget https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/zh_cn.csv -O /root/stable-diffusion-webui/extensions/tagcomplete/tags/zh_cn.csv

RUN https_proxy=${HTTP_PROXY} git clone https://github.com/opparco/stable-diffusion-webui-composable-lora.git  /root/stable-diffusion-webui/extensions/composable-lora
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/opparco/stable-diffusion-webui-two-shot.git  /root/stable-diffusion-webui/extensions/two-shot

RUN mkdir -p  /root/stable-diffusion-webui/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose
# RUN echo "{\"localization\": \"zh_CN\", \"control_net_max_models_num\": 3, \"tac_translation.translationFile\": \"zh_cn.csv\", \"sd_vae\": \"vae-ft-mse-840000-ema-pruned.ckpt\"}" > /root/stable-diffusion-webui/config.json
# CLIP 所需物料
RUN mkdir -p  /root/stable-diffusion-webui/stable-diffusion-webui/interrogate
RUN wget https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/artists.txt -O /root/stable-diffusion-webui/stable-diffusion-webui/interrogate/artists.txt
RUN wget https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/flavors.txt -O /root/stable-diffusion-webui/stable-diffusion-webui/interrogate/flavors.txt
RUN wget https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/mediums.txt -O /root/stable-diffusion-webui/stable-diffusion-webui/interrogate/mediums.txt
RUN wget https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/movements.txt -O /root/stable-diffusion-webui/stable-diffusion-webui/interrogate/movements.txt
RUN wget https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/configs/config.json -O /root/stable-diffusion-webui/config.json
# 下载模型(默认不下载)
#RUN cd  /root/stable-diffusion-webui/models/Stable-diffusion \
#    &&wget -nd -np -r  -c http://apksamba.ops.ilongyuan.cn:8000/ai/7/AI%E7%BE%8E%E6%9C%AF/%E6%89%93%E5%8C%85/models/Stable-diffusion/
#RUN cd /root/stable-diffusion-webui/  \
#    &&  python3 extensions/sd-webui-controlnet/install.py
# 确定OPEN_CLIP 和arkupsafe版本
RUN pip3 install basicsr Werkzeug==2.1.0 open_clip_torch==2.16.0 markupsafe==2.0.1 onnxruntime-gpu -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple
RUN pip3 uninstall gradio -y


WORKDIR /root/stable-diffusion-webui

# 启动时必须将models下面的一些必要文件挂载进去，例如：VAE-approx/models.pt
CMD bash -c "cd /root/stable-diffusion-webui; python3 -u webui.py --server-name 0.0.0.0 --api --clip-models-path=models/CLIP --xformers --enable-insecure-extension-access --no-half-vae --cors-allow-origins=\"*\">>nohup.out"