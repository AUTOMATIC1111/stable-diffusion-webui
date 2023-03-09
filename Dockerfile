# build cmd: docker build -t sd-webui . --build-arg HTTP_PROXY=http://172.16.255.22:3128
# FROM i.harbor.dragonest.net/xingzhe/sd-webui

# build dockerbase/Dockerfile first
# run cmd:
# nvidia-docker run -p 7860:7860 -it -v /data/apksamba/sd/models:/root/stable-diffusion-webui/models i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui:v0.7 /bin/bash
# nvidia-docker run -d -p 7860:7860 -v /data/apksamba/sd/models:/root/stable-diffusion-webui/models i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui:v0.8
FROM i.harbor.dragonest.net/xingzhe/sd-webui/sd-webui-env:v0.1

MAINTAINER wangdongming "wangdongming@dragonest.com"

ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

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

RUN python3 -V
#RUN python3 -c "import sys; print(sys.executable)"
RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple/


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
# xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.16rc425')
# gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
# clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
# openclip_package = os.environ.get('OPENCLIP_PACKAGE', "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b")
RUN cd ~/stable-diffusion-webui  \
    && pip3 install xformers -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple \
    && pip3 install -r requirements.txt -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple \
    && pip3 install -r requirements_versions.txt -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple
# repositories
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/CompVis/taming-transformers.git ~/stable-diffusion-webui/repositories/taming-transformers
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Stability-AI/stablediffusion.git ~/stable-diffusion-webui/repositories/stable-diffusion-stability-ai
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/crowsonkb/k-diffusion.git ~/stable-diffusion-webui/repositories/k-diffusion
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/sczhou/CodeFormer.git ~/stable-diffusion-webui/repositories/CodeFormer
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/salesforce/BLIP.git ~/stable-diffusion-webui/repositories/BLIP

RUN pip3 install setuptools_rust -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r  ~/stable-diffusion-webui/repositories/CodeFormer/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r  ~/stable-diffusion-webui/repositories/k-diffusion/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip3 install -r  ~/stable-diffusion-webui/repositories/BLIP/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN ls ~/stable-diffusion-webui/repositories/BLIP/
#RUN pip install -e  ~/stable-diffusion-webui/repositories/k-diffusion
RUN cd  ~/stable-diffusion-webui/repositories/stable-diffusion-stability-ai \
    && python3 setup.py install
#RUN ~/stable-diffusion-webui/repositories/taming-transformers  \
#    && python3 setup.py install
#RUN cd  ~/stable-diffusion-webui/repositories/BLIP/ && \
#    python3 setup.py install
# ControlNet
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/Mikubill/sd-webui-controlnet.git ~/stable-diffusion-webui/extensions/sd-webui-controlnet
RUN https_proxy=${HTTP_PROXY} git clone https://huggingface.co/webui/ControlNet-modules-safetensors ~/stable-diffusion-webui/models/ControlNet
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN ~/stable-diffusion-webui/extensions/stable-diffusion-webui-localization-zh_CN
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git ~/stable-diffusion-webui/extensions/tagger
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/yfszzx/stable-diffusion-webui-images-browser ~/stable-diffusion-webui/extensions/images-browser
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/fkunn1326/openpose-editor.git ~/stable-diffusion-webui/extensions/openpose-editor
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients ~/stable-diffusion-webui/extensions/aesthetic-gradients
RUN https_proxy=${HTTP_PROXY} git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen ~/stable-diffusion-webui/extensions/stable-diffusion-webui-promptgen
RUN mkdir -p  ~/stable-diffusion-webui/stable-diffusion-webui/extensions/sd-webui-controlnet/annotator/openpose
RUN echo "{\"localization\": \"zh_CN\"}" >  ~/stable-diffusion-webui/config.json

# 下载模型(默认不下载)
#RUN cd  ~/stable-diffusion-webui/models/Stable-diffusion \
#    &&wget -nd -np -r  -c http://apksamba.ops.ilongyuan.cn:8000/ai/7/AI%E7%BE%8E%E6%9C%AF/%E6%89%93%E5%8C%85/models/Stable-diffusion/
#RUN cd ~/stable-diffusion-webui/  \
#    &&  python3 extensions/sd-webui-controlnet/install.py
# 确定OPEN_CLIP 和arkupsafe版本
RUN pip3 install basicsr Werkzeug==2.1.0 open_clip_torch==2.16.0 markupsafe==2.0.1 -i https://nexus.ops.dragonest.com/repository/ly_pip_all/simple


WORKDIR ~/stable-diffusion-webui

# 启动时必须将models下面的一些必要文件挂载进去，例如：VAE-approx/models.pt
CMD bash -c "cd ~/stable-diffusion-webui; python3 -u webui.py --server-name 0.0.0.0 --xformers --enable-insecure-extension-access>>nohup.out"