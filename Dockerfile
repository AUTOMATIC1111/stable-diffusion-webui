FROM nvcr.io/nvidia/pytorch:22.11-py3

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive PIP_EXISTS_ACTION=w  PIP_PREFER_BINARY=1

RUN apt-get update && apt install -y software-properties-common && apt-get update && apt-get install -y python3.8-venv libxext6 libsm6 libgl1

WORKDIR stable-diffusion


COPY . .

RUN python -m venv venv --system-site-packages6