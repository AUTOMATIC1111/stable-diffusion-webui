FROM python:3.10

WORKDIR /webui

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 dos2unix google-perftools -y

COPY . .

RUN dos2unix ./webui.sh ./webui-user.sh

RUN groupadd --system --gid 1000 webui && \
    useradd webui --uid 1000 --gid 1000 --create-home --shell /bin/bash && \
    chown -R webui:webui .
USER 1000:1000

RUN ./webui.sh --exit --skip-torch-cuda-test

CMD [ "./webui.sh", "--skip-prepare-environment", "--listen" ]
