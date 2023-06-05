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
    "prompt": "city at night",
    "negative_prompt": "foggy, blurry",
    "steps": 20,
    "batch_size": 1,
    "n_iter": 1,
    "seed": -1,
    "sampler_name": "UniPC",
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

def generate(num: int = 0):
    log.info(f'sending generate request: {num+1} {options}')
    data = post('/sdapi/v1/txt2img', options)
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
