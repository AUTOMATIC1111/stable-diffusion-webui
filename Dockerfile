FROM python:3.10.9-bullseye
ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1
RUN mkdir /app
WORKDIR /app
COPY . .
RUN apt update && apt install libgl1 -y
RUN wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -P models/Stable-diffusion/
RUN pip install -r requirements.txt && pip install -r worker/requirements.txt && pip uninstall -y opencv-python && pip install opencv-python-headless triton
ENTRYPOINT ["python3", "launch.py", "--api", "--nowebui", "--listen"]
CMD ["--xformers", "--port", "7777"]
