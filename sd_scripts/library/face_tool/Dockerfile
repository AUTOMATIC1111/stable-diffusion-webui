FROM i.harbor.dragonest.net/public/python:3.10.13

LABEL authors="wangdongming@dragonest.com,wangying@dragonest.com,mawenbo@dragonest.com"

ARG TZ=Asia/Shanghai
ARG PROXY="http://172.16.241.101:8888"
WORKDIR  /sd-tool-server

COPY . /sd-tool-server
RUN mv /sd-tool-server/sources.list /etc/apt/  && rm /etc/apt/sources.list.d/debian.sources \
  && apt-get update \
  && apt-get install -yq  libgl1 \
  && cd /sd-tool-server \ 
  && pip install -r requirements.txt -i http://172.16.52.62:8081/repository/ly_pip_all/simple --trusted-host 172.16.52.62 --timeout=600 

EXPOSE 12305
CMD ["python3","main.py"]
