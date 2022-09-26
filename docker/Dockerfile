# DOCKERFILE CREATED BY COFFEEF0X, INTENDED FOR AUTOMATIC111'S STABLE DIFFUSION WEBUI
# FOLLOW COFFEEF0X @ HTTPS://GITLAB.COM/EXOFOX OR HTTPS://GITHUB.COM/COFFEEF0X
##########################################################################################
# IMPORTANT!!! SET BUILD CONTEXT TO ../ (PARENT DIRECTORY) AND LEAVE IN THE DOCKER
# SUBFOLDER, OTHERWISE THE FILE PATHS WILL NEED TO BE CORRECTED
#####
# THIS DOCKERFILE EXPECTS . . .
#####
# KNOWN LIMITATIONS:
### - CURRENTLY ONLY SUPPORTS NVIDIA GPUS, BUT CAN BE USED AS A STARTING POINT

FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

ARG CONFIG_DIRECTORY config-proxy
ARG STABLE_DIFFUSION_URL domain.example.com

ENV CONDA_DIR /root/anaconda
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

# INSTALL ANACONDA
# NOTE: MINICONDA COULD PROBABLY BE USED HERE FOR A PRODUCTION BUILD
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN chmod +x Anaconda3-2022.05-Linux-x86_64.sh
RUN ./Anaconda3-2022.05-Linux-x86_64.sh -b -p ${CONDA_DIR}
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

COPY models/sd-v1-4.ckpt ./model.ckpt
COPY ESRGAN/4xESRGAN.pth ./
COPY embeddings/ ./embeddings/

# SETUP DOCKER ENTRYPOINT
COPY docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# SETUP ANACONDA ENVIRONMENT FOR WEBUI
# NOTE: RUNNING AN VENV ENVIRONMENT IS UNECESSARY IN A PRODUCTION BUILD
RUN conda env create -f environment-wsl2.yaml
RUN conda init bash

SHELL ["conda", "run", "-n", "automatic", "/bin/bash", "-c"]

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

# RUN LAUNCH.PY WITH A TEMPORARY WEBUI.PY
RUN mv webui.py webui-8ef2cfb4-1648-464d-b1ee-f4068c17a36c.py
RUN echo 'print("launch successful!!!")' > webui.py
# RUN python launch.py
RUN rm -f webui.py
RUN mv webui-8ef2cfb4-1648-464d-b1ee-f4068c17a36c.py webui.py

# START THE SERVER
ENTRYPOINT [ "bash", "-i", "entrypoint.sh" ]

# TODO: ADJUST environment-wsl2.yaml TO USE PYTHON 3.10, RENAME TO environment.yaml, THEN UPDATE README + WIKI ACCORDINGLY (POSSIBLE ISSUE: CHANGING THE WSL2 ENVIRONMENT WILL REMOVE THE VENV SETUP)
# TODO: REPLACE ANACONDA WITH MINICONDA
# TODO: MOVE ENVIRONMENT OUTSIDE OF VENV
# TODO: MOVE MODEL DEPENDENCY FROM USER PROVIDED TO URL REFERENCED
# TODO: USE launch.py WITH A FAKE webui.py TO FETCH REMAINING DEPENDENCIES
# TODO: ADD ENV VARIABLE TO SKIP THE PULL OF THE ONLINE REPO AND USE CURRENT REPO, CAN BE USEFUL FOR DEVELOPMENT
# TODO: CHANGE FROM ROOT TO USER-BASED FOR SECURITY
# TODO: ADD README AND WIKI INFO FOR DOCKER