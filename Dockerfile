# FROM i.harbor.dragonest.net/xingzhe/sd-webui
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

MAINTAINER wangdongming "wangdongming@dragonest.com"

ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive

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
        python3-venv

RUN python3 -V
#RUN python3 -c "import sys; print(sys.executable)"



# 创建用户
#RUN groupadd --gid $USER_GID $USERNAME \
#       && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#       && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#       && chmod 0440 /etc/sudoers.d/$USERNAME

# 安装stable-diffusion-webui
# RUN pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN curl -sL https://raw.githubusercontent.com/Jackstrawcd/stable-diffusion-webui/master/webui.sh | sudo -u $USERNAME env COMMANDLINE_ARGS="${BUILD_ARGS}" bash \
#    && sudo -u $USERNAME python3 -m pip install xformers\
#    && sudo -u $USERNAME git clone https://github.com/Mikubill/sd-webui-controlnet.git /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet \
#    && sudo -u $USERNAME git clone https://huggingface.co/webui/ControlNet-modules-safetensors /home/$USERNAME/stable-diffusion-webui/models/ControlNet \
#    && sudo -u $USERNAME mkdir -p /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose \
#       && sudo -u $USERNAME curl -Lo /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose/body_pose_model.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth \
#       && sudo -u $USERNAME curl -Lo /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose/hand_pose_model.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth

#COPY --chown=$USERNAME:$USERNAME ${VOLUME}/config.json /home/$USERNAME/stable-diffusion-webui/config.json
#
#RUN \
#    sed -i -e 's/"outputs\//"\/'${VOLUME}'\/outputs\//g' -e 's/\(.*outdir_save.*:\ \).*/\1"\/'${VOLUME}'\/outputs\/save"/' /home/$USERNAME/stable-diffusion-webui/config.json
#
#WORKDIR /home/$USERNAME/stable-diffusion-webui
#USER $USERNAME
#
#CMD bash -c ". $HOME/stable-diffusion-webui/venv/bin/activate ; bash /${VOLUME}/linking.sh ; python3 -u webui.py --server-name 0.0.0.0 --xformers --enable-insecure-extension-access"
# 从GITHUB下载代码，在GITHUB托管方便合并最新代码

# RUN git config --global http.proxy $HTTP_PROXY
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Jackstrawcd/stable-diffusion-webui.git ~/stable-diffusion-webui

# 安装requirements
RUN cd ~/stable-diffusion-webui  \
    && pip3 install xformers -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install -r requirements_versions.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# repositories
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/CompVis/taming-transformers.git ~/stable-diffusion-webui/repositories/taming-transformers
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Stability-AI/stablediffusion.git ~/stable-diffusion-webui/repositories/stable-diffusion-stability-ai
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/crowsonkb/k-diffusion.git ~/stable-diffusion-webui/repositories/k-diffusion
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/sczhou/CodeFormer.git ~/stable-diffusion-webui/repositories/CodeFormer
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/salesforce/BLIP.git ~/stable-diffusion-webui/repositories/BLIP
# ControlNet
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Mikubill/sd-webui-controlnet.git ~/stable-diffusion-webui/extensions/sd-webui-controlnet
RUN https_proxy=${HTTP_PROXY} git clone https://huggingface.co/webui/ControlNet-modules-safetensors ~/stable-diffusion-webui/models/ControlNet
RUN mkdir -p  ~/stable-diffusion-webui/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose


#RUN cd ~/stable-diffusion-webui/  \
#    &&  python3 extensions/sd-webui-controlnet/install.py

WORKDIR ~/stable-diffusion-webui

CMD bash -c "cd ~/stable-diffusion-webui; python3 -u webui.py --server-name 0.0.0.0 --xformers --enable-insecure-extension-access"