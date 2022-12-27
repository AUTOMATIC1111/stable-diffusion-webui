FROM python:3.10.9-bullseye
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && pip uninstall -y opencv-python && pip install opencv-python-headless triton
RUN wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -P models/Stable-diffusion/
