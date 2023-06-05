#!/usr/bin/env python
# pylint: disable=no-member
"""generate batches of images from prompts and upscale them

params: run with `--help`

default workflow runs infinite loop and prints stats when interrupted:
1. choose random scheduler lookup all available and pick one
2. generate dynamic prompt based on styles, embeddings, places, artists, suffixes
3. beautify prompt
4. generate 3x3 images
5. create image grid
6. upscale images with face restoration
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import math
import os
import pathlib
import secrets
import time
import sys
import importlib

from random import randrange
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import ImageFileDirectory_v2

from sdapi import close, get, interrupt, post, session
from util import Map, log, safestring


sd = {}
random = {}
stats = Map({ 'images': 0, 'wall': 0, 'generate': 0, 'upscale': 0 })
avg = {}


def grid(data):
    if len(data.image) > 1:
        w, h = data.image[0].size
        rows = round(math.sqrt(len(data.image)))
        cols = math.ceil(len(data.image) / rows)
        image = Image.new('RGB', size = (cols * w, rows * h), color = 'black')
        for i, img in enumerate(data.image):
            image.paste(img, box=(i % cols * w, i // cols * h))
        short = data.info.prompt[:min(len(data.info.prompt), 96)] # limit prompt part of filename to 96 chars
        name = '{seed:0>9} {short}'.format(short = short, seed = data.info.all_seeds[0]) # pylint: disable=consider-using-f-string
        name = safestring(name) + '.jpg'
        f = os.path.join(sd.paths.root, sd.paths.grid, name)
        log.info({ 'grid': { 'name': f, 'size': image.size, 'images': len(data.image) } })
        image.save(f, 'JPEG', exif = exif(data.info, None, 'grid'), optimize = True, quality = 70)
        return image


def exif(info, i = None, op = 'generate'):
    seed = [info.all_seeds[i]] if len(info.all_seeds) > 0 and i is not None else info.all_seeds # always returns list
    seed = ', '.join([str(x) for x in seed]) # int list to str list to single str
    template = '{prompt} | negative {negative_prompt} | seed {s} | steps {steps} | cfgscale {cfg_scale} | sampler {sampler_name} | batch {batch_size} | timestamp {job_timestamp} | model {model} | vae {vae}'.format(s = seed, model = sd.options['sd_model_checkpoint'], vae = sd.options['sd_vae'], **info) # pylint: disable=consider-using-f-string
    if op == 'upscale':
        template += ' | faces gfpgan' if sd.upscale.gfpgan_visibility > 0 else ''
        template += ' | faces codeformer' if sd.upscale.codeformer_visibility > 0 else ''
        template += ' | upscale {resize}x {upscaler}'.format(resize = sd.upscale.upscaling_resize, upscaler = sd.upscale.upscaler_1) if sd.upscale.upscaler_1 != 'None' else '' # pylint: disable=consider-using-f-string
        template += ' | upscale {resize}x {upscaler}'.format(resize = sd.upscale.upscaling_resize, upscaler = sd.upscale.upscaler_2) if sd.upscale.upscaler_2 != 'None' else '' # pylint: disable=consider-using-f-string
    if op == 'grid':
        template += ' | grid {num}'.format(num = sd.generate.batch_size * sd.generate.n_iter) # pylint: disable=consider-using-f-string
    ifd = ImageFileDirectory_v2()
    exif_stream = io.BytesIO()
    _TAGS = dict(((v, k) for k, v in TAGS.items())) # enumerate possible exif tags
    ifd[_TAGS['ImageDescription']] = template
    ifd.save(exif_stream)
    val = b'Exif\x00\x00' + exif_stream.getvalue()
    return val


def randomize(lst):
    if len(lst) > 0:
        return secrets.choice(lst)
    else:
        return ''


def prompt(params): # generate dynamic prompt or use one if provided
    sd.generate.prompt = params.prompt if params.prompt != 'dynamic' else randomize(random.prompts)
    sd.generate.negative_prompt = params.negative if params.negative != 'dynamic' else randomize(random.negative)
    embedding = params.embedding if params.embedding != 'random' else randomize(random.embeddings)
    sd.generate.prompt = sd.generate.prompt.replace('<embedding>', embedding)
    artist = params.artist if params.artist != 'random' else randomize(random.artists)
    sd.generate.prompt = sd.generate.prompt.replace('<artist>', artist)
    style = params.style if params.style != 'random' else randomize(random.styles)
    sd.generate.prompt = sd.generate.prompt.replace('<style>', style)
    suffix = params.suffix if params.suffix != 'random' else randomize(random.suffixes)
    sd.generate.prompt = sd.generate.prompt.replace('<suffix>', suffix)
    place = params.suffix if params.suffix != 'random' else randomize(random.places)
    sd.generate.prompt = sd.generate.prompt.replace('<place>', place)
    if params.prompts or params.debug:
        log.info({ 'random initializers': random })
    if params.prompt == 'dynamic':
        log.info({ 'dynamic prompt': sd.generate.prompt })
    return sd.generate.prompt


def sampler(params, options): # find sampler
    if params.sampler == 'random':
        sd.generate.sampler_name = randomize(options.samplers)
        log.info({ 'random sampler': sd.generate.sampler_name })
    else:
        found = [i for i in options.samplers if i.startswith(params.sampler)]
        if len(found) == 0:
            log.error({ 'sampler error': sd.generate.sampler_name, 'available': options.samplers})
            exit()
        sd.generate.sampler_name = found[0]
    return sd.generate.sampler_name


async def generate(prompt = None, options = None, quiet = False): # pylint: disable=redefined-outer-name
    global sd # pylint: disable=global-statement
    if options:
        sd = Map(options)
    if prompt is not None:
        sd.generate.prompt = prompt
    if not quiet:
        log.info({ 'generate': sd.generate })
    names = []
    b64s = []
    images = []
    info = Map({})
    data = await post('/sdapi/v1/txt2img', sd.generate)
    if 'error' in data:
        log.error({ 'generate': data['error'], 'reason': data['reason'] })
        return Map({})
    info = Map(json.loads(data['info']))
    log.debug({ 'info': info })
    images = data['images']
    short = info.prompt[:min(len(info.prompt), 96)] # limit prompt part of filename to 64 chars
    for i in range(len(images)):
        b64s.append(images[i])
        images[i] = Image.open(io.BytesIO(base64.b64decode(images[i].split(',',1)[0])))
        name = '{seed:0>9} {short}'.format(short = short, seed = info.all_seeds[i]) # pylint: disable=consider-using-f-string
        name = safestring(name) + '.jpg'
        f = os.path.join(sd.paths.root, sd.paths.generate, name)
        names.append(f)
        if not quiet:
            log.info({ 'image': { 'name': f, 'size': images[i].size } })
        images[i].save(f, 'JPEG', exif = exif(info, i), optimize = True, quality = 70)
    return Map({ 'name': names, 'image': images, 'b64': b64s, 'info': info })


async def upscale(data):
    data.upscaled = []
    if sd.upscale.upscaling_resize <=1:
        return data
    sd.upscale.image = ''
    log.info({ 'upscale': sd.upscale })
    for i in range(len(data.image)):
        f = data.name[i].replace(sd.paths.generate, sd.paths.upscale)
        sd.upscale.image = data.b64[i]
        res = await post('/sdapi/v1/extra-single-image', sd.upscale)
        image = Image.open(io.BytesIO(base64.b64decode(res['image'].split(',',1)[0])))
        data.upscaled.append(image)
        log.info({ 'image': { 'name': f, 'size': image.size } })
        image.save(f, 'JPEG', exif = exif(data.info, i, 'upscale'), optimize = True, quality = 70)
    return data


async def init():
    '''
    import torch
    log.info({ 'torch': torch.__version__, 'available': torch.cuda.is_available() })
    current_device = torch.cuda.current_device()
    mem_free, mem_total = torch.cuda.mem_get_info()
    log.info({ 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'arch': torch.cuda.get_arch_list(), 'device': torch.cuda.get_device_name(current_device), 'memory': { 'free': round(mem_free / 1024 / 1024), 'total': (mem_total / 1024 / 1024) } })
    '''
    options = Map({})
    options.flags = await get('/sdapi/v1/cmd-flags')
    log.debug({ 'flags': options.flags })
    data = await get('/sdapi/v1/sd-models')
    options.models = [obj['title'] for obj in data]
    log.debug({ 'registered models': options.models })
    found = sd.options.sd_model_checkpoint if sd.options.sd_model_checkpoint in options.models else None
    if found is None:
        found = [i for i in options.models if i.startswith(sd.options.sd_model_checkpoint)]
    if len(found) == 0:
        log.error({ 'model error': sd.generate.sd_model_checkpoint, 'available': options.models})
        exit()
    sd.options.sd_model_checkpoint = found[0]
    data = await get('/sdapi/v1/samplers')
    options.samplers = [obj['name'] for obj in data]
    log.debug({ 'registered samplers': options.samplers })
    data = await get('/sdapi/v1/upscalers')
    options.upscalers = [obj['name'] for obj in data]
    log.debug({ 'registered upscalers': options.upscalers })
    data = await get('/sdapi/v1/face-restorers')
    options.restorers = [obj['name'] for obj in data]
    log.debug({ 'registered face restorers': options.restorers })
    await interrupt()
    await post('/sdapi/v1/options', sd.options)
    options.options = await get('/sdapi/v1/options')
    log.info({ 'target models': { 'diffuser': options.options['sd_model_checkpoint'], 'vae': options.options['sd_vae'] } })
    log.info({ 'paths': sd.paths })
    options.queue = await get('/queue/status')
    log.info({ 'queue': options.queue })
    pathlib.Path(sd.paths.root).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(sd.paths.root, sd.paths.generate)).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(sd.paths.root, sd.paths.upscale)).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(sd.paths.root, sd.paths.grid)).mkdir(parents = True, exist_ok = True)
    return options


def args(): # parse cmd arguments
    global sd # pylint: disable=global-statement
    global random # pylint: disable=global-statement
    parser = argparse.ArgumentParser(description = 'sd pipeline')
    parser.add_argument('--config', type = str, default = 'generate.json', required = False, help = 'configuration file')
    parser.add_argument('--random', type = str, default = 'random.json', required = False, help = 'prompt file with randomized sections')
    parser.add_argument('--max', type = int, default = 1, required = False, help = 'maximum number of generated images')
    parser.add_argument('--prompt', type = str, default = 'dynamic', required = False, help = 'prompt')
    parser.add_argument('--negative', type = str, default = 'dynamic', required = False, help = 'negative prompt')
    parser.add_argument('--artist', type = str, default = 'random', required = False, help = 'artist style, used to guide dynamic prompt when prompt is not provided')
    parser.add_argument('--embedding', type = str, default = 'random', required = False, help = 'use embedding, used to guide dynamic prompt when prompt is not provided')
    parser.add_argument('--style', type = str, default = 'random', required = False, help = 'image style, used to guide dynamic prompt when prompt is not provided')
    parser.add_argument('--suffix', type = str, default = 'random', required = False, help = 'style suffix, used to guide dynamic prompt when prompt is not provided')
    parser.add_argument('--place', type = str, default = 'random', required = False, help = 'place locator, used to guide dynamic prompt when prompt is not provided')
    parser.add_argument('--faces', default = False, action='store_true', help = 'restore faces during upscaling')
    parser.add_argument('--steps', type = int, default = 0, required = False, help = 'number of steps')
    parser.add_argument('--batch', type = int, default = 0, required = False, help = 'batch size, limited by gpu vram')
    parser.add_argument('--n', type = int, default = 0, required = False, help = 'number of iterations')
    parser.add_argument('--cfg', type = int, default = 0, required = False, help = 'classifier free guidance scale')
    parser.add_argument('--sampler', type = str, default = 'random', required = False, help = 'sampler')
    parser.add_argument('--seed', type = int, default = 0, required = False, help = 'seed, default is random')
    parser.add_argument('--upscale', type = int, default = 0, required = False, help = 'upscale factor, disabled if 0')
    parser.add_argument('--model', type = str, default = '', required = False, help = 'diffusion model')
    parser.add_argument('--vae', type = str, default = '', required = False, help = 'vae model')
    parser.add_argument('--path', type = str, default = '', required = False, help = 'output path')
    parser.add_argument('--width', type = int, default = 0, required = False, help = 'width')
    parser.add_argument('--height', type = int, default = 0, required = False, help = 'height')
    parser.add_argument('--beautify', default = False, action='store_true', help = 'beautify prompt')
    parser.add_argument('--prompts', default = False, action='store_true', help = 'print dynamic prompt templates')
    parser.add_argument('--debug', default = False, action='store_true', help = 'print extra debug information')
    params = parser.parse_args()
    if params.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': params.__dict__ })
    home = pathlib.Path(sys.argv[0]).parent
    if os.path.isfile(params.config):
        try:
            with open(params.config, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sd = Map(data)
                log.debug({ 'config': sd })
        except Exception as e:
            log.error({ 'config error': params.config, 'exception': e })
            exit()
    elif os.path.isfile(os.path.join(home, params.config)):
        try:
            with open(os.path.join(home, params.config), 'r', encoding='utf-8') as f:
                data = json.load(f)
                sd = Map(data)
                log.debug({ 'config': sd })
        except Exception as e:
            log.error({ 'config error': params.config, 'exception': e })
            exit()
    else:
        log.error({ 'config file not found': params.config})
        exit()
    if params.prompt == 'dynamic':
        log.info({ 'prompt template': params.random })
        if os.path.isfile(params.random):
            try:
                with open(params.random, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    random = Map(data)
                    log.debug({ 'random template': sd })
            except:
                log.error({ 'random template error': params.random})
                exit()
        elif os.path.isfile(os.path.join(home, params.random)):
            try:
                with open(os.path.join(home, params.random), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    random = Map(data)
                    log.debug({ 'random template': sd })
            except:
                log.error({ 'random template error': params.random})
                exit()
        else:
            log.error({ 'random template file not found': params.random})
            exit()
        _dynamic = prompt(params)

    sd.paths.root = params.path if params.path != '' else sd.paths.root
    sd.generate.restore_faces = params.faces if params.faces is not None else sd.generate.restore_faces
    sd.generate.seed = params.seed if params.seed > 0 else sd.generate.seed
    sd.generate.sampler_name = params.sampler if params.sampler != 'random' else sd.generate.sampler_name
    sd.generate.batch_size = params.batch if params.batch > 0 else sd.generate.batch_size
    sd.generate.cfg_scale = params.cfg if params.cfg > 0 else sd.generate.cfg_scale
    sd.generate.n_iter = params.n if params.n > 0 else sd.generate.n_iter
    sd.generate.width = params.width if params.width > 0 else sd.generate.width
    sd.generate.height = params.height if params.height > 0 else sd.generate.height
    sd.generate.steps = params.steps if params.steps > 0 else sd.generate.steps
    sd.upscale.upscaling_resize = params.upscale if params.upscale > 0 else sd.upscale.upscaling_resize
    sd.upscale.codeformer_visibility = 1 if params.faces else sd.upscale.codeformer_visibility
    sd.options.sd_vae = params.vae if params.vae != '' else sd.options.sd_vae
    sd.options.sd_model_checkpoint = params.model if params.model != '' else sd.options.sd_model_checkpoint
    sd.upscale.upscaler_1 = 'SwinIR_4x' if params.upscale > 1 else sd.upscale.upscaler_1
    if sd.generate.cfg_scale == 0:
        sd.generate.cfg_scale = randrange(5, 10)
    return params


async def main():
    params = args()
    sess = await session()
    if sess is None:
        await close()
        exit()
    options = await init()
    iteration = 0
    while True:
        iteration += 1
        log.info('')
        log.info({ 'iteration': iteration, 'batch': sd.generate.batch_size, 'n': sd.generate.n_iter, 'total': sd.generate.n_iter * sd.generate.batch_size })
        dynamic = prompt(params)
        if params.beautify:
            try:
                promptist = importlib.import_module('modules.promptist')
                sd.generate.prompt = promptist.beautify(dynamic)
            except Exception as e:
                log.error({ 'beautify': e })
        scheduler = sampler(params, options)
        t0 = time.perf_counter()
        data = await generate() # generate returns list of images
        if not 'image' in data:
            break
        stats.images += len(data.image)
        t1 = time.perf_counter()
        if len(data.image) > 0:
            avg[scheduler] = (t1 - t0) / len(data.image)
        stats.generate += t1 - t0
        _image = grid(data)
        data = await upscale(data)
        t2 = time.perf_counter()
        stats.upscale += t2 - t1
        stats.wall += t2 - t0
        its = sd.generate.steps / ((t1 - t0) / len(data.image)) if len(data.image) > 0 else 0
        avg_time = round((t1 - t0) / len(data.image)) if len(data.image) > 0 else 0
        log.info({ 'time' : { 'wall': round(t1 - t0), 'average': avg_time, 'upscale': round(t2 - t1), 'its': round(its, 2) } })
        log.info({ 'generated': stats.images, 'max': params.max, 'progress': round(100 * stats.images / params.max, 1) })
        if params.max != 0 and stats.images >= params.max:
            break


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        asyncio.run(interrupt())
        asyncio.run(close())
        log.info({ 'interrupt': True })
    finally:
        log.info({ 'sampler performance': avg })
        log.info({ 'stats' : stats })
        asyncio.run(close())
'''
    except Exception as e:
        log.info({ 'sampler performance': avg })
        log.info({ 'stats': stats })
        log.critical({ 'exception': e })
        exit()
'''
