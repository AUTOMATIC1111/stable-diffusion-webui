FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ARG APP_PATH=/app
ARG REPO_PATH=$APP_PATH/repositories

COPY --chown=1001:1001 ./ $APP_PATH
WORKDIR $APP_PATH

ARG DEBIAN_FRONTEND=noninteractive
USER root
RUN apt-get update -yq
RUN apt-get upgrade -yq
RUN apt-get install -yq git python3-pip libgl1 libglib2.0-0
RUN mkdir -p /.local /.config
RUN chown -R 1001:1001 /.local /.config

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r requirements.txt
USER 1001

RUN git clone https://github.com/CompVis/stable-diffusion.git $REPO_PATH/stable-diffusion && \
    git -C $REPO_PATH/stable-diffusion checkout 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc

RUN git clone https://github.com/CompVis/taming-transformers.git $REPO_PATH/taming-transformers && \
    git -C $REPO_PATH/taming-transformers checkout 24268930bf1dce879235a7fddd0b2355b84d7ea6

RUN git clone https://github.com/salesforce/BLIP.git $REPO_PATH/BLIP && \
    git -C $REPO_PATH/BLIP checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9

RUN git clone https://github.com/sczhou/CodeFormer.git $REPO_PATH/CodeFormer && \
    git -C $REPO_PATH/CodeFormer checkout c5b4593074ba6214284d6acd5f1719b6c5d739af

USER root
RUN pip install -r repositories/CodeFormer/requirements.txt
USER 1001

ENTRYPOINT [ "python3", "webui.py", "--listen" ]
