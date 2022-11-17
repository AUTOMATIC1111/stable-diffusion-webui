FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip
ENV ENV prod
RUN apt update
RUN apt install -y git
RUN pip install --upgrade pip
RUN pip install cython
RUN pip install numpy
RUN pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
RUN pip install cmake
RUN pip install pycocotools
RUN pip install dlib
RUN pip install psutil
RUN pip install setuptools wheel
COPY . ./
RUN pip install font-roboto
RUN pip install -r requirements.txt
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 webui:app --timeout 540