FROM nvidia/cuda:12.2.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]


ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0
RUN apt-get clean -y && rm -rf /var/lib/apt/lists/*

# user and workdir
RUN useradd -m user
RUN mkdir /work && chown -R user:user /work
USER user
WORKDIR /work


################################
# setup
################################

RUN git clone https://github.com/webcoderz/stable-diffusion-webui.git
WORKDIR /work/stable-diffusion-webui



# setup
RUN python3 -mvenv venv && /work/stable-diffusion-webui/venv/bin/python -c "from launch import *; prepare_environment()" --skip-torch-cuda-test --no-download-sd-model



################################
# entrypoint
################################




EXPOSE 7860
EXPOSE 8000
EXPOSE 8265
EXPOSE 6388
EXPOSE 10001

CMD ["./webui.sh", "--xformers --cors-allow-origins=* --api"]