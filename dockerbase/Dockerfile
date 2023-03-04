FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

MAINTAINER wangdongming "wangdongming@dragonest.com"
ENV TERM linux
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE DontWarn
ENV DEBIAN_FRONTEND noninteractive

ARG TZ=Asia/Shanghai
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
RUN apt-get install -yq --no-install-recommends python3.10 python3-venv

CMD python -V