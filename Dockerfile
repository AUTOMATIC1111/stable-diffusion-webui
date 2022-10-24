FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt update \
  && apt-key adv --fetch-keys \
  && apt install --no-install-recommends -y build-essential wget git curl unzip python3 python3-venv python3-pip libgl1 libglib2.0-0 \
  && apt clean && rm -rf /var/lib/apt/lists/*

# Optional, install (as root) only if you want to download models and embeddings from S3
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && ./aws/install

RUN useradd -ms /bin/bash sd
USER sd

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /home/sd/stable-diffusion-webui
RUN mkdir repositories

RUN  git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion && \
  cd repositories/stable-diffusion && git checkout 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc && \
  python3 setup.py install --user

RUN git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers && \
  cd repositories/taming-transformers && git checkout 24268930bf1dce879235a7fddd0b2355b84d7ea6

RUN git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer && \
  cd repositories/CodeFormer && git checkout c5b4593074ba6214284d6acd5f1719b6c5d739af

RUN git clone https://github.com/salesforce/BLIP.git repositories/BLIP && \
  cd repositories/BLIP && git checkout 48211a1594f1321b00f14c9f7a5b4813144b2fb9

RUN git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion && \
  cd repositories/k-diffusion && git checkout f4e99857772fc3a126ba886aadf795a332774878

RUN git clone https://github.com/Hafiidz/latent-diffusion repositories/latent-diffusion && \
  pip install -r repositories/latent-diffusion/requirements.txt --prefer-binary

# install requirements of Stable Diffusion
RUN pip install transformers==4.19.2 diffusers invisible-watermark --prefer-binary

# install k-diffusion
RUN pip install git+https://github.com/crowsonkb/k-diffusion.git --prefer-binary

# (optional) install GFPGAN (face restoration)
RUN pip install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 --prefer-binary

# (optional) install requirements for CodeFormer (face restoration)
RUN pip install -r repositories/CodeFormer/requirements.txt --prefer-binary

# update numpy to latest version
RUN pip install -U numpy  --prefer-binary

RUN pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 --prefer-binary

COPY --chown=sd ./ .
RUN chmod a=rwx entrypoint.sh
RUN pip install -r requirements.txt  --prefer-binary

ENTRYPOINT ./entrypoint.sh
