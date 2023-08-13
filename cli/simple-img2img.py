#!/usr/bin/env python
import os
import io
import sys
import base64
import logging
import requests
import urllib3
from PIL import Image

sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
options = {
    "init_images": [],
    "prompt": "city at night",
    "negative_prompt": "foggy, blurry",
    "steps": 20,
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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{sd_url}{endpoint}', json = dct, timeout=300, verify=False, auth=auth())
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
    options['init_images'] = [encode('html/logo-dark.png')]
    options['batch_size'] = len(options['init_images'])
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
