#!/usr/bin/env python
# pylint: disable=no-member
import os
import re
import json
import time
import logging
import importlib
import asyncio
import argparse
from pathlib import Path
from util import Map, log
from sdapi import get, post, close
from generate import generate # pylint: disable=import-error
grid = importlib.import_module('image-grid').grid


options = Map({
    # used by extra networks
    'prompt': 'photo of <keyword> <embedding>, photograph, posing, pose, high detailed, intricate, elegant, sharp focus, skin texture, looking forward, facing camera, 135mm, shot on dslr, canon 5d, 4k, modelshoot style, cinematic lighting',
    # used by models
    'prompts': [
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
    ],
    # save format
    'format': '.jpg',
    # used by generate script
    'paths': {
        "root": "/mnt/c/Users/mandi/OneDrive/Generative/Generate",
        "generate": "image",
        "upscale": "upscale",
        "grid": "grid",
    },
    # generate params
    'generate': {
        'restore_faces': True,
        'prompt': '',
        'negative_prompt': 'foggy, blurry, blurred, duplicate, ugly, mutilated, mutation, mutated, out of frame, bad anatomy, disfigured, deformed, censored, low res, low resolution, watermark, text, poorly drawn face, poorly drawn hands, signature',
        'steps': 20,
        'batch_size': 2,
        'n_iter': 1,
        'seed': -1,
        'sampler_name': 'UniPC',
        'cfg_scale': 6,
        'width': 512,
        'height': 512,
    },
    'lora': {
        'strength': 1.0,
    },
    'hypernetwork': {
        'keyword': '',
        'strength': 1.0,
    },
})


def preview_exists(folder, model):
    model = os.path.splitext(model)[0]
    for suffix in ['', '.preview']:
        for ext in ['.jpg', '.png', '.webp']:
            fn = os.path.join(folder, f'{model}{suffix}{ext}')
            if os.path.exists(fn):
                return True
    return False


async def preview_models(params):
    data = await get('/sdapi/v1/sd-models')
    allmodels = [m['title'] for m in data]
    models = []
    excluded = []
    for m in allmodels: # loop through all registered models
        ok = True
        for e in params.exclude: # check if model is excluded
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
    opt = await get('/sdapi/v1/options')
    log.info({ 'total jobs': len(models) * options.generate.batch_size, 'per-model': options.generate.batch_size })
    log.info(json.dumps(options, indent=2))
    for model in models:
        if preview_exists(opt['ckpt_dir'], model) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'model preview exists': model })
            continue
        fn = os.path.join(opt['ckpt_dir'], os.path.splitext(model)[0] + options.format)
        log.info({ 'model load': model })

        opt['sd_model_checkpoint'] = model
        del opt['sd_lora']
        del opt['sd_lyco']
        await post('/sdapi/v1/options', opt)
        opt = await get('/sdapi/v1/options')
        images = []
        labels = []
        t0 = time.time()
        for label, p in options.prompts:
            options.generate.prompt = p
            log.info({ 'model generating': model, 'label': label, 'prompt': options.generate.prompt })
            data = await generate(options = options, quiet=True)
            if 'image' in data:
                for img in data['image']:
                    images.append(img)
                    labels.append(label)
            else:
                log.error({ 'model': model, 'error': data })
        t1 = time.time()
        if len(images) == 0:
            log.error({ 'model': model, 'error': 'no images generated' })
            continue
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'model preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })

    opt = await get('/sdapi/v1/options')
    if opt['sd_model_checkpoint'] != params.model:
        log.info({ 'model set default': params.model })
        opt['sd_model_checkpoint'] = params.model
        del opt['sd_lora']
        del opt['sd_lyco']
        await post('/sdapi/v1/options', opt)


async def lora(params):
    opt = await get('/sdapi/v1/options')
    folder = opt['lora_dir']
    if not os.path.exists(folder):
        log.error({ 'lora directory not found': folder })
        return
    models1 = list(Path(folder).glob('**/*.safetensors'))
    models2 = list(Path(folder).glob('**/*.ckpt'))
    models = [os.path.splitext(f)[0] for f in models1 + models2]
    log.info({ 'loras': len(models) })
    for model in models:
        if preview_exists('', model) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'lora preview exists': model })
            continue
        fn = model + options.format
        model = os.path.basename(model)
        images = []
        labels = []
        t0 = time.time()
        keywords = re.sub(r'\d', '', model)
        keywords = keywords.replace('-v', ' ').replace('-', ' ').strip().split(' ')
        keyword = '\"' + '\" \"'.join(keywords) + '\"'
        options.generate.prompt = options.prompt.replace('<keyword>', keyword)
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
        if len(images) == 0:
            log.error({ 'model': model, 'error': 'no images generated' })
            continue
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'lora preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def lyco(params):
    opt = await get('/sdapi/v1/options')
    folder = opt['lyco_dir']
    if not os.path.exists(folder):
        log.error({ 'lyco directory not found': folder })
        return
    models1 = list(Path(folder).glob('**/*.safetensors'))
    models2 = list(Path(folder).glob('**/*.ckpt'))
    models = [os.path.splitext(f)[0] for f in models1 + models2]
    log.info({ 'lycos': len(models) })
    for model in models:
        if preview_exists('', model) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'lyco preview exists': model })
            continue
        fn = model + options.format
        model = os.path.basename(model)
        images = []
        labels = []
        t0 = time.time()
        keywords = re.sub(r'\d', '', model)
        keywords = keywords.replace('-v', ' ').replace('-', ' ').strip().split(' ')
        keyword = '\"' + '\" \"'.join(keywords) + '\"'
        options.generate.prompt = options.prompt.replace('<keyword>', keyword)
        options.generate.prompt = options.generate.prompt.replace('<embedding>', '')
        options.generate.prompt += f' <lyco:{model}:{options.lora.strength}>'
        log.info({ 'lyco generating': model, 'keyword': keyword, 'prompt': options.generate.prompt })
        data = await generate(options = options, quiet=True)
        if 'image' in data:
            for img in data['image']:
                images.append(img)
                labels.append(keyword)
        else:
            log.error({ 'lyco': model, 'keyword': keyword, 'error': data })
        t1 = time.time()
        if len(images) == 0:
            log.error({ 'model': model, 'error': 'no images generated' })
            continue
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'lyco preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def hypernetwork(params):
    opt = await get('/sdapi/v1/options')
    folder = opt['hypernetwork_dir']
    if not os.path.exists(folder):
        log.error({ 'hypernetwork directory not found': folder })
        return
    models = [os.path.splitext(f)[0] for f in Path(folder).glob('**/*.pt')]
    log.info({ 'hypernetworks': len(models) })
    for model in models:
        if preview_exists(folder, model) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'hypernetwork preview exists': model })
            continue
        fn = os.path.join(folder, model + options.format)
        images = []
        labels = []
        t0 = time.time()
        keyword = options.hypernetwork.keyword
        options.generate.prompt = options.prompt.replace('<keyword>', options.hypernetwork.keyword)
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
        if len(images) == 0:
            log.error({ 'model': model, 'error': 'no images generated' })
            continue
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'hypernetwork preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def embedding(params):
    opt = await get('/sdapi/v1/options')
    folder = opt['embeddings_dir']
    if not os.path.exists(folder):
        log.error({ 'embeddings directory not found': folder })
        return
    models = [os.path.splitext(f)[0] for f in Path(folder).glob('**/*.pt')]
    log.info({ 'embeddings': len(models) })
    for model in models:
        if preview_exists(folder, model) and len(params.input) == 0: # if model preview exists and not manually included
            log.info({ 'embedding preview exists': model })
            continue
        fn = os.path.join(folder, model + '.preview' + options.format)
        images = []
        labels = []
        t0 = time.time()
        keyword = '\"' + re.sub(r'\d', '', model) + '\"'
        options.generate.batch_size = 4
        options.generate.prompt = options.prompt.replace('<keyword>', keyword)
        options.generate.prompt = options.generate.prompt.replace('<embedding>', '')
        log.info({ 'embedding generating': model, 'keyword': keyword, 'prompt': options.generate.prompt })
        data = await generate(options = options, quiet=True)
        if 'image' in data:
            for img in data['image']:
                images.append(img)
                labels.append(keyword)
        else:
            log.error({ 'embeding': model, 'keyword': keyword, 'error': data })
        t1 = time.time()
        if len(images) == 0:
            log.error({ 'model': model, 'error': 'no images generated' })
            continue
        image = grid(images = images, labels = labels, border = 8)
        log.info({ 'saving preview': fn, 'images': len(images), 'size': [image.width, image.height] })
        image.save(fn)
        t = t1 - t0
        its = 1.0 * options.generate.steps * len(images) / t
        log.info({ 'embeding preview created': model, 'image': fn, 'images': len(images), 'grid': [image.width, image.height], 'time': round(t, 2), 'its': round(its, 2) })


async def create_previews(params):
    await preview_models(params)
    await lora(params)
    await lyco(params)
    await hypernetwork(params)
    await embedding(params)
    await close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'generate model previews')
    parser.add_argument('--model', default='best/icbinp-icantbelieveIts-final.safetensors [73f48afbdc]', help="model used to create extra network previews")
    parser.add_argument('--exclude', default=['sd-v20', 'sd-v21', 'inpainting', 'pix2pix'], help="exclude models with keywords")
    parser.add_argument('--debug', default = False, action='store_true', help = 'print extra debug information')
    parser.add_argument('input', type = str, nargs = '*')
    args = parser.parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': args.__dict__ })
    asyncio.run(create_previews(args))
