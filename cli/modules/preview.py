#!/bin/env python
import io
import json
import base64
import logging
from PIL import Image

from util import Map, log
from sdapi import getsync, postsync

mask = 'preview-body.jpg'
template = 'photo of "{name}", {suffix}, high detailed, skin texture, facing camera, 135mm, shot on dslr, 4k, modelshoot style'
img2img_options = {
    'prompt': None,
    'negative_prompt': '',
    'init_images': [],
    'sampler_name': 'DPM2 Karras',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 30,
    'cfg_scale': 6,
    'width': 512,
    'height': 512,
    'restore_faces': False
}

def encode(f):
    img = Image.open(f)
    with io.BytesIO() as stream:
        img.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded

def img2img(name: str, suffix: str):
    options = getsync('/sdapi/v1/options')
    img2img_options['prompt'] = template.format(name = name, suffix = suffix)
    log.info({ 'preview prompt': img2img_options['prompt'] })
    log.info({ 'preview options': img2img_options })
    img2img_options['init_images'].append(encode(mask))
    data = postsync('/sdapi/v1/img2img', img2img_options)
    if 'error' in data:
        log.error({ 'preview': data['error'], 'reason': data['reason'] })
        return
    info = Map(json.loads(data['info']))
    log.debug({ 'preview info': info })
    if not 'images' in data:
        log.error({ 'preview': 'no images' })
        return
    obj = data.copy()
    del obj['images']
    log.info({ 'preview': { 'model': options["sd_model_checkpoint"], 'seed': obj['info']['seed'] } })
    image = Image.open(io.BytesIO(base64.b64decode(data['images'][0].split(",",1)[0])))
    image.save('test.jpg')

if __name__ == "__main__":
    # log.setLevel(logging.DEBUG)
    log.info({ 'preview': 'start' })
    img2img('hanna', 'person, woman, girl, model')
