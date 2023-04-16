#!/bin/env python
"""
create preview images from embeddings
"""
import os
import io
import sys
import json
import base64
import argparse
from pathlib import Path
from PIL import Image
from inspect import getsourcefile
from util import Map, log
from sdapi import getsync, postsync
from grid import grid

template = 'photo of "{name}", {suffix}, high detailed, skin texture, looking forward, facing camera, 135mm, shot on dslr, 4k, modelshoot style'
img2img_options = Map({
    'prompt': None,
    'negative_prompt': 'cartoon, drawing, cgi, sketch, comic, disfigured, deformed',
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
    img2img_options['prompt'] = template.format(name = name, suffix = suffix)
    log.debug({ 'preview options': img2img_options })
    if len(img2img_options['init_images']) == 0:
        for i in range(img2img_options.batch_size):
            mask = os.path.join(os.path.dirname(getsourcefile(lambda:0)), 'preview-template'+ str(i+1) +'.jpg')
            if (not os.path.isfile(mask)):
                log.error({ 'preview': 'missing preview mask' })
                return
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

    parser = argparse.ArgumentParser(description = 'generate embeddings previews')
    parser.add_argument('--overwrite', default = False, action='store_true', help = 'overwrite existing previews')
    parser.add_argument('input', type=str, nargs='*')
    params = parser.parse_args()

    if len(params.input) == 0:
        files = list(Path(cmdflags.embeddings_dir).glob('*.pt'))
    else:
        files = list(os.path.join(cmdflags.embeddings_dir, a + '.pt') for a in params.input if os.path.isfile(os.path.join(cmdflags.embeddings_dir, a + '.pt')))
    candidates = [str(f) for f in files]
    candidates.sort(key=os.path.getctime, reverse=True)

    files = []
    for f in candidates:
        fn = f.replace('.pt', '.preview.png')
        if os.path.isfile(f.replace('.pt', '.preview.png')):
            if params.overwrite:
                log.info({ 'preview add': fn })
                files.append(f)
            else:
                log.info({ 'preview skip': fn })
        else:
            log.info({ 'preview add': fn })
            files.append(f)

    log.info({ 'preview embeddings': len(files) })
    for f in files:
        name = Path(f).stem
        create_preview(name, 'person')
