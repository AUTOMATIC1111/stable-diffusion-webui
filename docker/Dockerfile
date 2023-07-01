FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 libglib2.0-0 \
	python3 python3-venv \
	git \
	wget \
	vim \
	inetutils-ping \
	sudo \
	net-tools \
	iproute2 \
	&& \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY webui.sh webui.sh

# setup venv in /venv to avoid conflict with volume in /stable-diffusion-webui
RUN echo 'venv_dir=/venv' > webui-user.sh

ENV install_dir=/
RUN ./webui.sh -f can_run_as_root --exit --skip-torch-cuda-test

ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR "/stable-diffusion-webui/"
VOLUME /root/.cache

CMD ["python3", "launch.py", "--listen"]
