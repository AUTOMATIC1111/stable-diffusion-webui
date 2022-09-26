##########################################################################################
# IMPORTANT!!! THE BUILD CONTEXT MUST BE SET TO THE ../ (REPOSITORY ROOT DIRECTORY)
# AND EXPECTS DOCKER IN ITS SUBFOLDER, OTHERWISE THE FILE PATHS MUST BE MODIFIED
#####
# THIS DOCKERFILE CURRENTLY EXPECTS THE USER TO PROVIDE THE FOLLOWING
# - THE STABLE DIFFUSION MODEL IN/AS -> models/model.ckpt
# - THE ESRGAN MODEL IN/AS -> ESRGAN/4xESRGAN.pth
#####
# KNOWN LIMITATIONS:
### - CURRENTLY ONLY SUPPORTS NVIDIA GPUS, BUT CAN BE USED AS A STARTING POINT

FROM debian:11-slim

ENV DEBIAN_FRONTEND noninteractive

ARG CONFIG_DIRECTORY config-proxy
ARG STABLE_DIFFUSION_URL domain.example.com

ENV CONDA_DIR /root/conda
ENV SD_ROOT_DIR /sd

ENV SD_USERNAME sa
ENV SD_PASSWORD password
ENV COMMANDLINE_ARGS --ui-config-file "config/ui-config.json" --ui-settings-file "config/ui-settings.json" --listen
# IF YOU PLAN ON HOSTING THIS PUBLICLY, PLEASE ADD A LOGIN SCREEN BY APPENDING THE ARG BELOW TO THE COMMANDLINE_ARGS ABOVE
# --gradio-auth ${SD_USERNAME}:${SD_PASSWORD}

# IF YOU WOULD LIKE TO USE CODE FROM THE WEBUI, YOU CAN USE --allow-code
# FOR A LIST OF MORE OPTIONS, PLEASE REFER TO https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/shared.py
# MANY OF THESE OPTIONS CAN BREAK THIS CONTAINER'S STRUCTURE, USE AT YOUR OWN RISK!!!

EXPOSE 7860
WORKDIR ${SD_ROOT_DIR}

SHELL ["/bin/bash", "-c"]

# UPDATE UBUNTU, INSTALL DEPENDENCIES, THEN CLEAN THE REMAINING PACKAGES
RUN apt-get update
RUN apt-get install -y libglib2.0-0 wget git 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# INSTALL MINICONDA
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
RUN chmod +x Miniconda3-py39_4.12.0-Linux-x86_64.sh
RUN ./Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p ${CONDA_DIR}
ENV PATH ${CONDA_DIR}/bin:$PATH

# INSTALL A CLEAN AND UPDATED REPO OF THE WEBUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
RUN rm -rf stable-diffusion-webui/.git

# SET WORKDIR TO THE DOWNLOADED REPO
WORKDIR ${SD_ROOT_DIR}/stable-diffusion-webui

# DOWNLOAD WEBUI MODELS AND REPOSITORIES
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

RUN mkdir repositories
RUN git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
RUN git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers
RUN git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
RUN git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# IMPORT EXPECTED DATASETS AND ALL PROVIDED EMBEDDINGS
COPY models/model.ckpt ./model.ckpt
COPY ESRGAN/4xESRGAN.pth ./
COPY embeddings/ ./embeddings/

# SETUP ANACONDA ENVIRONMENT FOR WEBUI
RUN conda install python=3.10.4
RUN conda install pytorch==1.11.0 -c pytorch
RUN conda install torchvision==0.12.0 -c pytorch
RUN conda install cudatoolkit=11.3 -c pytorch
RUN conda clean -a -y

RUN pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary
RUN pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary
RUN pip install git+https://github.com/TencentARC/GFPGAN.git --prefer-binary
RUN pip install -r repositories/CodeFormer/requirements.txt --prefer-binary
RUN pip install -r requirements.txt --prefer-binary
RUN pip install -U numpy --prefer-binary
RUN pip install --prefer-binary -U opencv-python-headless

# SETUP REVERSE PROXY, MOVE DEFAULT CONF FILE TO DIRECTORY FOR NGINX CONFIG MOUNT
# NOT THE CLEANEST SOLUTION, BUT PROBABLY THE SIMPLEST FOR THE END USER
# MAKES THE REVERSE PROXY DEPENDENT ON THE STABLE DIFFUSION CONTAINER
COPY ./config-proxy/domain.example.com ./config-proxy/${STABLE_DIFFUSION_URL}

# RUN LAUNCH.PY WITH A TEMPORARY WEBUI.PY,
# THIS WILL ENSURE MOST DOWNLOADS BECOME A PART OF THE IMAGE
# ALLOWING RESTARTS TO BE MUCH QUICKER AND WITH MORE EFFICIENT BANDWIDTH USE
RUN mv webui.py webui-8ef2cfb4-1648-464d-b1ee-f4068c17a36c.py
RUN echo 'def webui():' > webui.py
RUN echo '  print("launch successful!!!")' >> webui.py
RUN COMMANDLINE_ARGS="${COMMANDLINE_ARGS} --skip-torch-cuda-test" && python launch.py
RUN rm -f webui.py
RUN mv webui-8ef2cfb4-1648-464d-b1ee-f4068c17a36c.py webui.py

# START THE SERVER
ENTRYPOINT [ "python", "launch.py" ]

# TODO: REALIZED THAT THE MODELS ARE PULLED FROM LATEST, MAKING THE IMAGE VERY UNSTABLE. 
#       IT MAY BE BEST TO LAUNCH PY EARLIER TO PULL REPO VERSIONS THE AUTOMATIC111 HAS TESTED.
# TODO: MOVE MODEL DEPENDENCY FROM USER PROVIDED TO URL REFERENCED
# TODO: ADD ENV VARIABLE TO SKIP THE PULL OF THE ONLINE REPO AND USE CURRENT REPO, CAN BE USEFUL FOR DEVELOPMENT
# TODO: CHANGE FROM ROOT TO USER-BASED FOR SECURITY
# TODO: ADD README AND WIKI INFO FOR DOCKER