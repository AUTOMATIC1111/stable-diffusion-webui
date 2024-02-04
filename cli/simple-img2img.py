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

filename='/tmp/simple-img2img.jpg'
options = {
    "save_images": False,
    "send_images": True,
}


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
        image.close()
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def generate(args): # pylint: disable=redefined-outer-name
    t0 = time.time()
    if args.model is not None:
        post('/sdapi/v1/options', { 'sd_model_checkpoint': args.model })
        post('/sdapi/v1/reload-checkpoint') # needed if running in api-only to trigger new model load
    options['prompt'] = args.prompt
    options['negative_prompt'] = args.negative
    options['steps'] = int(args.steps)
    options['seed'] = int(args.seed)
    options['sampler_name'] = args.sampler
    options['init_images'] = [encode(args.init)]
    image = Image.open(args.init)
    options['width'] = image.width
    options['height'] = image.height
    image.close()
    if args.mask is not None:
        options['mask'] = encode(args.mask)
    data = post('/sdapi/v1/img2img', options)
    t1 = time.time()
    if 'images' in data:
        for i in range(len(data['images'])):
            b64 = data['images'][i].split(',',1)[0]
            info = data['info']
            image = Image.open(io.BytesIO(base64.b64decode(b64)))
            image.save(filename)
            log.info(f'received image: size={image.size} file={filename} time={t1-t0:.2f} info="{info}"')
    else:
        log.warning(f'no images received: {data}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'simple-img2img')
    parser.add_argument('--init', required=True, help='init image')
    parser.add_argument('--mask', required=False, help='mask image')
    parser.add_argument('--prompt', required=False, default='', help='prompt text')
    parser.add_argument('--negative', required=False, default='', help='negative prompt text')
    parser.add_argument('--steps', required=False, default=20, help='number of steps')
    parser.add_argument('--seed', required=False, default=-1, help='initial seed')
    parser.add_argument('--sampler', required=False, default='Euler a', help='sampler name')
    parser.add_argument('--model', required=False, help='model name')
    args = parser.parse_args()
    log.info(f'img2img: {args}')
    generate(args)
