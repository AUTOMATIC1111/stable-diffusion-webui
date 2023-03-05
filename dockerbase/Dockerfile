FROM ubuntu:22.04

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
	    python3 \
	    python3-pip \
	    python3-venv \
	    python3-opencv

CMD python3 -c "import torch; print(torch.cuda.is_available())"