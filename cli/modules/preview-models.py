#!/bin/env python
import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from util import Map, log
from sdapi import get, post, close
from grid import grid

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from generate import sd, generate


default = 'sd-v15-runwayml.ckpt [cc6cb27103]'
exclude = ['sd-v20', 'sd-v21', 'inpainting', 'pix2pix']
# used by lora
prompt = "photo of <keyword> <embedding>, photograph, posing, pose, high detailed, intricate, elegant, sharp focus, skin texture, looking forward, facing camera, 135mm, shot on dslr, canon 5d, 4k, modelshoot style, cinematic lighting"

# used by models
prompts = [
    ('photo citiscape', 'cityscape during night, photorealistic, high detailed, sharp focus, depth of field, 4k'),
    ('photo car', 'photo of a sports car, high detailed, sharp focus, dslr, cinematic lighting, realistic'),
    ('photo woman', 'portrait photo of beautiful woman, high detailed, dslr, 35mm'),
    ('photo naked', 'full body photo of beautiful sexy naked woman, high detailed, dslr, 35mm'),

    ('photo taylor', 'portrait photo of beautiful woman taylor swift, high detailed, sharp focus, depth of field, dslr, 35mm <lora:taylor-swift:1>'),
    ('photo ti-mia', 'portrait photo of beautiful woman "ti-mia", naked, high detailed, dslr, 35mm'),
    ('photo ti-vlado', 'portrait photo of man "ti-vlado", high detailed, dslr, 35mm'),
    ('photo lora-vlado', 'portrait photo of man vlado, high detailed, dslr, 35mm <lora:vlado-original:1>'),

    ('wlop', 'a stunning portrait of sexy teen girl in a wet t-shirt, vivid color palette, digital painting, octane render, highly detailed, particles, light effect, volumetric lighting, art by wlop'),
    ('greg rutkowski', 'beautiful woman, high detailed, sharp focus, depth of field, 4k, art by greg rutkowski'),
    ('carne griffiths', 'beautiful woman taylor swift, high detailed, sharp focus, depth of field, art by carne griffiths <lora:taylor-swift:1>'),
    ('carne griffiths', 'man vlado, high detailed, sharp focus, depth of field, art by carne griffiths <lora:vlado-full:1>'),
]

options = Map({
    'generate': {
        'restore_faces': True,
        'prompt': '',
        'negative_prompt': 'digital art, cgi, render, foggy, blurry, blurred, duplicate, ugly, mutilated, mutation, mutated, out of frame, bad anatomy, disfigured, deformed, censored, low res, low resolution, watermark, text, poorly drawn face, poorly drawn hands, signature',
        'steps': 20,
        'batch_size': 2,
        'n_iter': 1,
        'seed': -1,
        'sampler_name': 'UniPC',
        'cfg_scale': 6,
        'width': 512,
        'height': 512,
    },
    'format': '.jpg',
    'paths': {
        "root": "/mnt/c/Users/mandi/OneDrive/Generative/Generate",
        "generate": "image",
        "upscale": "upscale",
        "grid": "grid",
    },
    'options': {
        "sd_model_checkpoint": "sd-v15-runwayml",
        "sd_vae": "vae-ft-mse-840000-ema-pruned.ckpt",
    },
    'lora': {
        'strength': 0.9,
    },
    'hypernetwork': {
        'keyword': 'beautiful sexy woman',
        'strength': 1.0,
    },
})


async def models(params):
    global sd
    data = await get('/sdapi/v1/sd-models')
    all = [m['title'] for m in data]
    models = []
    excluded = []
    for m in all: # loop through all registered models
        ok = True
        for e in exclude: # check if model is excluded
            if e in m:
                excluded.append(m)
                ok = False
                break
        if ok:
            short = m.split(' [')[0]
            short = short.replace('.ckpt', '').replace('.safetensors', '')
            models.append(short)
    if len(params.input) > 0: # check if model is included in cmd line
        filtered = []
        for m in params.input:
            if m in models:
                filtered.append(m)
            else:
                log.error({ 'model not found': m })
                return
        models = filtered
    log.info({ 'models preview' })
    log.info({ 'models': len(models), 'excluded': len(excluded) })
    cmdflags = await get('/sdapi/v1/cmd-flags')
    opt = await get('/sdapi/v1/options')
    if params.output != '':
        dir = params.output
    else:
        dir = os.path.abspath(os.path.join(cmdflags['hypernetwork_dir'], '..', 'Stable-diffusion'))
    log.info({ 'output directory': dir })
    log.info({ 'total jobs': len(models) * options.generate.batch_size, 'per-model': options.generate.batch_size })
    log.info(json.dumps(options, indent=2))
    for model in models:
        fn = os.path.join(dir, os.path.basename(model) + options.format)
        if os.path.exists(fn) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'model preview exists': model })
            continue
        log.info({ 'model load': model })

        opt['sd_model_checkpoint'] = model
        await post('/sdapi/v1/options', opt)
        opt = await get('/sdapi/v1/options')
        images = []
        labels = []
        t0 = time.time()
        for label, prompt in prompts:
            options.generate.prompt = prompt
            log.info({ 'model generating': model, 'label': label, 'prompt': options.generate.prompt })
            data = await generate(options = options, quiet=True)
            if 'image' in data:
                for img in data['image']:
                    images.append(img)
                    labels.append(label)
            else:
                log.error({ 'model': model, 'error': data })
        t1 = time.time()
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'model preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })
    
    opt = await get('/sdapi/v1/options')
    if opt['sd_model_checkpoint'] != default and not params.fixed:
        log.info({ 'model set default': default })
        opt['sd_model_checkpoint'] = default
        await post('/sdapi/v1/options', opt)


async def lora(params):
    cmdflags = await get('/sdapi/v1/cmd-flags')
    dir = cmdflags['lora_dir']
    if not os.path.exists(dir):
        log.error({ 'lora directory not found': dir })
        return
    models1 = [f for f in Path(dir).glob('*.safetensors')]
    models2 = [f for f in Path(dir).glob('*.ckpt')]
    models = [f.stem for f in models1 + models2]
    log.info({ 'loras': len(models) })
    for model in models:
        fn = os.path.join(dir, model + options.format)
        if os.path.exists(fn) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'lora preview exists': model })
            continue
        images = []
        labels = []
        t0 = time.time()
        import re
        keywords = re.sub('\d', '', model)
        keywords = keywords.replace('-v', ' ').replace('-', ' ').strip().split(' ')
        keyword = '\"' + '\" \"'.join(keywords) + '\"'
        options.generate.prompt = prompt.replace('<keyword>', keyword)
        options.generate.prompt = options.generate.prompt.replace('<embedding>', '')
        options.generate.prompt += f' <lora:{model}:{options.lora.strength}>'
        log.info({ 'lora generating': model, 'keyword': keyword, 'prompt': options.generate.prompt })
        data = await generate(options = options, quiet=True)
        if 'image' in data:
            for img in data['image']:
                images.append(img)
                labels.append(keyword)
        else:
            log.error({ 'lora': model, 'keyword': keyword, 'error': data })
        t1 = time.time()
        image = grid(images = images, labels = labels, border = 8)
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'lora preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def hypernetwork(params):
    cmdflags = await get('/sdapi/v1/cmd-flags')
    dir = cmdflags['hypernetwork_dir']
    if not os.path.exists(dir):
        log.error({ 'hypernetwork directory not found': dir })
        return
    models = [f.stem for f in Path(dir).glob('*.pt')]
    log.info({ 'loras': len(models) })
    for model in models:
        fn = os.path.join(dir, model + options.format)
        if os.path.exists(fn) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'hypernetwork preview exists': model })
            continue
        images = []
        labels = []
        t0 = time.time()
        keyword = options.hypernetwork.keyword
        options.generate.prompt = prompt.replace('<keyword>', options.hypernetwork.keyword)
        options.generate.prompt = options.generate.prompt.replace('<embedding>', '')
        options.generate.prompt = f' <hypernet:{model}:{options.hypernetwork.strength}> ' + options.generate.prompt
        log.info({ 'hypernetwork generating': model, 'keyword': keyword, 'prompt': options.generate.prompt })
        data = await generate(options = options, quiet=True)
        if 'image' in data:
            for img in data['image']:
                images.append(img)
                labels.append(keyword)
        else:
            log.error({ 'hypernetwork': model, 'keyword': keyword, 'error': data })
        t1 = time.time()
        image = grid(images = images, labels = labels, border = 8)
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'hypernetwork preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def create_previews(params):
    await models(params)
    await lora(params)
    await hypernetwork(params)
    await close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'generate model previews')
    parser.add_argument('--output', type = str, default = '', required = False, help = 'output directory')
    parser.add_argument('--fixed', default = False, action='store_true', help = "do not change model")
    parser.add_argument('input', type = str, nargs = '*')
    params = parser.parse_args()
    asyncio.run(create_previews(params))
