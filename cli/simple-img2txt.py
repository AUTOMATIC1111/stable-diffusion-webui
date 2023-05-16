#!/usr/bin/env python
import io
import sys
import base64
import logging
import requests
from PIL import Image

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
sd_url = "http://127.0.0.1:7860"
options = {
    "init_images": [],
    "prompt": "city at night",
    "negative_prompt": "foggy, blurry",
    "steps": 1,
    "batch_size": 1,
    "n_iter": 1,
    "seed": -1,
    "sampler_name": "Euler a",
    "cfg_scale": 6,
    "width": 512,
    "height": 512,
    "save_images": False,
    "send_images": True,
}

def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300)
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()

def encode(f):
    image = Image.open(f)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded

def generate(num: int = 0):
    log.info(f'sending generate request: {num+1} {options}')
    options['init_images'] = [encode('../html/logo.png')]
    data = post('/sdapi/v1/img2img', options)
    if 'images' in data:
        for i in range(len(data['images'])):
            b64 = data['images'][i].split(',',1)[0]
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            log.info(f'received image: {image.size}')
    else:
        log.warning(f'no images received: {data}')

if __name__ == "__main__":
    sys.argv.pop(0)
    repeats = int(''.join(sys.argv) or '1')
    log.info(f'repeats: {repeats}')
    for n in range(repeats):
        generate(n)
