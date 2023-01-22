#!/bin/env python
import os
import io
import json
import base64
from pathlib import Path
from PIL import Image

from util import Map, log
from sdapi import getsync, postsync
from grid import grid

# masks = ['preview-face.jpg', 'preview-body.jpg']
mask = 'preview-body.jpg'
template = 'photo of "{name}", {suffix}, high detailed, skin texture, facing camera, 135mm, shot on dslr, 4k, modelshoot style'
img2img_options = Map({
    'prompt': None,
    'negative_prompt': '',
    'init_images': [],
    'sampler_name': 'DPM2 Karras',
    'batch_size': 4,
    'n_iter': 1,
    'steps': 30,
    'cfg_scale': 6,
    'width': 512,
    'height': 512,
    'restore_faces': False
})

def encode(f):
    img = Image.open(f)
    with io.BytesIO() as stream:
        img.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded

def create_preview(name: str, suffix: str):
    options = getsync('/sdapi/v1/options')
    cmdflags = getsync('/sdapi/v1/cmd-flags')    
    print(cmdflags.embeddings_dir)
    img2img_options['prompt'] = template.format(name = name, suffix = suffix)
    log.info({ 'preview prompt': img2img_options['prompt'] })
    log.debug({ 'preview options': img2img_options })
    if len(img2img_options['init_images']) == 0:
        for i in range(img2img_options.batch_size):
            img2img_options['init_images'].append(encode(mask))
    data = postsync('/sdapi/v1/img2img', img2img_options)
    if 'error' in data:
        log.error({ 'preview': data['error'], 'reason': data['reason'] })
        return
    info = Map(json.loads(data['info']))
    if not 'images' in data:
        log.error({ 'preview': 'no images' })
        return
    fn = os.path.join(cmdflags.embeddings_dir, name + '.preview.png')
    log.info({ 'preview': { 'name': fn, 'model': options.sd_model_checkpoint, 'seed': info.seed } })
    images = []
    for b64 in data['images']:
        images.append(Image.open(io.BytesIO(base64.b64decode(b64.split(",",1)[0]))))
    image = grid(images, None, square=True)
    image.save(fn)

if __name__ == "__main__":
    log.info({ 'preview': 'start' })
    cmdflags = getsync('/sdapi/v1/cmd-flags')
    for f in Path(cmdflags.embeddings_dir).glob('*.pt'):
        create_preview(f.stem, 'person')
