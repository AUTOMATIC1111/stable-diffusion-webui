#!/usr/bin/env python
import os
import io
import time
import base64
import logging
import argparse
import requests
import urllib3
from PIL import Image

sd_url = os.environ.get('SDAPI_URL', "http://127.0.0.1:7860")
sd_username = os.environ.get('SDAPI_USR', None)
sd_password = os.environ.get('SDAPI_PWD', None)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def auth():
    if sd_username is not None and sd_password is not None:
        return requests.auth.HTTPBasicAuth(sd_username, sd_password)
    return None


def get(endpoint: str, dct: dict = None):
    req = requests.get(f'{sd_url}{endpoint}', json=dct, timeout=300, verify=False, auth=auth())
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


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
    log.info(f'encoding image: {image}')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def upscale(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    # options['mask'] = encode(args.mask)
    upscalers = get('/sdapi/v1/upscalers')
    upscalers = [u['name'] for u in upscalers]
    log.info(f'upscalers: {upscalers}')
    options = {
        "save_images": False,
        "send_images": True,
        'image': encode(args.input),
        'upscaler_1': args.upscaler,
        'resize_mode': 0, # rescale_by
        'upscaling_resize': args.scale,

    }
    data = post('/sdapi/v1/extra-single-image', options)
    t1 = time.time()
    if 'image' in data:
        b64 = data['image'].split(',',1)[0]
        image = Image.open(io.BytesIO(base64.b64decode(b64)))
        image.save(args.output)
        log.info(f'received: image={image} file={args.output} time={t1-t0:.2f}')
    else:
        log.warning(f'no images received: {data}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'simple-upscale')
    parser.add_argument('--input', required=True, help='input image')
    parser.add_argument('--output', required=True, help='output image')
    parser.add_argument('--upscaler', required=False, default='Nearest', help='upscaler name')
    parser.add_argument('--scale', required=False, default=2, help='upscaler scale')
    args = parser.parse_args()
    log.info(f'upscale: {args}')
    upscale(args)
