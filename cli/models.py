#!/bin/env python
import os
import sys
import json
import asyncio
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from modules.util import Map, log
from modules.sdapi import get, post, close
from generate import sd, generate
from modules.grid import grid


embeddings = ['blonde', 'bruntette', 'sexy', 'mia', 'lin', 'kelly', 'hanna', 'rreid-random-v0']
exclude = ['sd-v20', 'sd-v21', 'inpainting']
prompt = "photo of beautiful woman <embedding>, photograph, posing, pose, high detailed, intricate, elegant, sharp focus, skin texture, looking forward, facing camera, 135mm, shot on dslr, canon 5d, 4k, modelshoot style, cinematic lighting"
options = Map({
    'generate': {
        'restore_faces': True,
        'prompt': '',
        'negative_prompt': 'digital art, cgi, render, foggy, blurry, blurred, duplicate, ugly, mutilated, mutation, mutated, out of frame, bad anatomy, disfigured, deformed, censored, low res, low resolution, watermark, text, poorly drawn face, poorly drawn hands, signature',
        'steps': 30,
        'batch_size': 4,
        'n_iter': 1,
        'seed': -1,
        'sampler_name': 'DPM2 Karras',
        'cfg_scale': 7,
        'width': 512,
        'height': 512
    },
    'paths': {
        "root": "/mnt/c/Users/mandi/OneDrive/Generative/Generate",
        "generate": "image",
        "upscale": "upscale",
        "grid": "grid"
    },
    'options': {
        "sd_model_checkpoint": "sd-v15-runwayml",
        "sd_vae": "vae-ft-mse-840000-ema-pruned.ckpt"
    }
})


async def models(params):
    global sd
    data = await get('/sdapi/v1/sd-models')
    all = [m['title'] for m in data]
    models = []
    excluded = []
    for m in all:
        ok = True
        for e in exclude:
            if e in m:
                excluded.append(m)
                ok = False
                break
        if ok:
            models.append(m)
    log.info({ 'models preview' })
    log.info({ 'models': len(models), 'excluded': len(excluded) })
    log.info({ 'embeddings': len(embeddings) })
    log.info({ 'batch size': options.generate.batch_size })
    log.info({ 'total jobs': len(models) * len(embeddings) * options.generate.batch_size })
    log.info(json.dumps(options, indent=2))
    models = ['sd-v15-runwayml.ckpt [cc6cb27103]']
    for model in models:
        opt = await get('/sdapi/v1/options')
        opt['sd_model_checkpoint'] = model
        await post('/sdapi/v1/options', opt)
        images = []
        labels = []
        for embedding in embeddings:
            options.generate.prompt = prompt.replace('<embedding>', f'\"{embedding}\"')
            log.info({ 'embedding': embedding, 'prompt': options.generate.prompt })
            data = await generate(options = options)
            if 'image' in data:
                for img in data['image']:
                    images.append(img)
                    labels.append(embedding)
            else:
                log.error({ 'model': model, 'embedding': embedding, 'error': data })
        image = grid(images = images, labels = labels, border = 8)
        fn = os.path.join(params.output, model + '.jpg')
        image.save(fn)
        log.info({ 'file': fn, 'model': model, 'images': len(images), 'grid': [image.width, image.height] })
    await close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'generate model previews')
    parser.add_argument('--output', type = str, default = '', required = False, help = 'output directory')
    params = parser.parse_args()
    asyncio.run(models(params))
