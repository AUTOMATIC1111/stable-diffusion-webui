FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

MAINTAINER wangdongming "wangdongming@dragonest.com"

ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive

ARG TZ=Asia/Shanghai
ARG BUILD_ARGS

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
	    libglib2.0-0
RUN apt-get install -yq --no-install-recommends \
	    python3-opencv
RUN   apt-get install -yq --no-install-recommends python3.10 python3-venv
RUN python3 -V
RUN python3 -c "import sys; print(sys.executable)"
RUN bash -c " conda env list; conda init"

# 创建用户
RUN groupadd --gid $USER_GID $USERNAME \
       && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
       && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
       && chmod 0440 /etc/sudoers.d/$USERNAME


# 安装stable-diffusion-webui
RUN curl -sL https://raw.githubusercontent.com/Jackstrawcd/stable-diffusion-webui/master/webui.sh | sudo -u $USERNAME env COMMANDLINE_ARGS="${BUILD_ARGS}" bash \
    && sudo -u $USERNAME python3 -m pip install xformers \
    && sudo -u $USERNAME git clone https://github.com/Mikubill/sd-webui-controlnet.git /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet \
    && sudo -u $USERNAME git clone https://huggingface.co/webui/ControlNet-modules-safetensors /home/$USERNAME/stable-diffusion-webui/models/ControlNet \
    && sudo -u $USERNAME mkdir -p /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose \
       && sudo -u $USERNAME curl -Lo /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose/body_pose_model.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth \
       && sudo -u $USERNAME curl -Lo /home/$USERNAME/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose/hand_pose_model.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth

COPY --chown=$USERNAME:$USERNAME ${VOLUME}/config.json /home/$USERNAME/stable-diffusion-webui/config.json

RUN \
    sed -i -e 's/"outputs\//"\/'${VOLUME}'\/outputs\//g' -e 's/\(.*outdir_save.*:\ \).*/\1"\/'${VOLUME}'\/outputs\/save"/' /home/$USERNAME/stable-diffusion-webui/config.json

WORKDIR /home/$USERNAME/stable-diffusion-webui
USER $USERNAME

CMD bash -c ". $HOME/stable-diffusion-webui/venv/bin/activate ; bash /${VOLUME}/linking.sh ; python3 -u webui.py --server-name 0.0.0.0 --xformers --enable-insecure-extension-access"
