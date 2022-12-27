FROM python:3.10.9-bullseye
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
RUN wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
COPY . .
RUN mv v1-5-pruned-emaonly.ckpt models/Stable-diffusion/v1-5-pruned-emaonly.ckpt
RUN pip install -r requirements.txt && pip uninstall opencv-python && pip opencv-python-headless triton
