#!/bin/env python
# pylint: disable=no-member
"""
simple implementation of training api: `/sdapi/v1/train`
- supports: create embedding, image preprocess, train embedding (with all known parameters)
- does not (yet) support: create hyper-network, train hyper-network
- compatible with progress api: `/sdapi/v1/progress`
- if interrupted, auto-continues from last known step
- create and preprocess executed as sync jobs
- train is executed as async job with progress monitoring
"""

import argparse
import asyncio
import logging
import math
import os
import sys
import time
import json
from pathlib import Path, PurePath

import filetype
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from modules.util import Map, log
from modules.losschart import plot
from modules.ffmpeg import extract
from modules.lossrate import gen_loss_rate_str
from modules.sdapi import close, get, interrupt, post, progress, session
from modules.process import process_images
from modules.grid import grid


images = []
args = {}
options = None
cmdflags = None


async def plotloss(params):
    logdir = os.path.abspath(os.path.join(cmdflags['embeddings_dir'], '../train/log'))
    try:
        plot(logdir, params.name)
    except Exception as err:
        log.warning({ 'loss chart error': err })


async def captions(docs: list):
    exclude = ['a', 'in', 'on', 'out', 'at', 'the', 'and', 'with', 'next', 'to', 'it', 'for', 'of', 'into', 'that']
    d = dict()
    for f in docs:
        text = open(f, 'r', encoding='utf-8')
        for line in text:
            line = line.strip()
            line = line.lower()
            words = line.split(" ")
            for word in words:
                if word in exclude:
                    continue
                d[word] = d[word] + 1 if word in d else 1
    pairs = ((value, key) for (key,value) in d.items())
    sort = sorted(pairs, reverse = True)
    if len(sort) > 10:
        del sort[10:]
    d = {k: v for v, k in sort}
    log.info({ 'top captions': d })


async def cleanup(params):
    if params.nocleanup:
        return
    log.info({ 'preprocess cleanup': params.dst })
    for f in Path(params.dst).glob('*.png'):
        f.unlink()
    for f in Path(params.dst).glob('*.txt'):
        f.unlink()


async def preprocess_builtin(params):
    global images # pylint: disable=global-statement
    log.debug({ 'preprocess start' })
    if os.path.isdir(params.dst):
        if params.overwrite:
            log.info({ 'preprocess deleting existing images': params.dst })
            for f in Path(params.dst).glob('*.png'):
                f.unlink()
            for f in Path(params.dst).glob('*.txt'):
                f.unlink()
        else:
            log.error({ 'preprocess output folder already exists': params.dst })
            return 0
    files = [os.path.join(params.src, f) for f in os.listdir(params.src) if os.path.isfile(os.path.join(params.src, f))]
    candidates = [f for f in files if filetype.is_image(f)]
    not_images = [f for f in files if (not filetype.is_image(f) and not f.endswith('.txt'))]
    images = []
    low_res = []
    for f in candidates:
        img = Image.open(f)
        mp = (img.size[0] * img.size[1]) / 1024 / 1024
        if mp < 1 or img.size[0] < 512 or img.size[1] < 512:
            low_res.append(f)
            os.rename(f, f + '.skip')
        else:
            images.append(f)
    log.debug({ 'preprocess skipping': not_images })
    log.debug({ 'preprocess low res': low_res })
    args.preprocess.process_src = params.src
    args.preprocess.process_dst = params.dst
    log.debug({ 'preprocess args': args.preprocess })
    _res = await post('/sdapi/v1/preprocess', json = args.preprocess)
    processed = [os.path.join(params.dst, f) for f in os.listdir(params.dst) if os.path.isfile(os.path.join(params.dst, f))]
    processed_imgs = [f for f in processed if f.endswith('.png')]
    processed_docs = [f for f in processed if f.endswith('.txt')]
    log.info({ 'preprocess': {
        'source': params.src,
        'destination': params.dst,
        'files': len(files),
        'images': len(images),
        'processed': len(processed_imgs),
        'captions': len(processed_docs),
        'skipped': len(not_images),
        'low-res': len(low_res) }
    })
    if len(processed_docs) > 0:
        await captions(processed_docs)
    return len(processed_imgs)


async def preprocess(params):
    global images # pylint: disable=global-statement
    res = 0
    if os.path.isfile(params.src):
        if not filetype.is_video(params.src):
            kind = filetype.guess(params.src)
            log.error({ 'preprocess error': { 'not a valid movie file': params.src, 'guess': kind } })
        else:
            extract_dst = os.path.join(params.dst, 'extract')
            log.debug({ 'preprocess args': args.extract_video })
            images = extract(params.src, extract_dst, rate = args.extract_video.rate, fps = args.extract_video.fps, start = args.extract_video.vstart, end = args.extract_video.vend) # extract keyframes from movie
            if images > 0:
                params.src = extract_dst
                res = await preprocess(params) # call again but now with keyframes
            else:
                log.error({ 'preprocess video extract': 'no images' })
    elif os.path.isdir(params.src):
        if params.preprocess == 'builtin':
            res = preprocess_builtin(params)
        if params.preprocess == 'custom':
            t0 = time.perf_counter()
            args.preprocess.process_src = params.src
            args.preprocess.process_dst = params.dst
            process_images(src = params.src, dst = params.dst)
            i = [os.path.join(params.dst, f) for f in os.listdir(params.dst) if os.path.isfile(os.path.join(params.dst, f)) and filetype.is_image(os.path.join(params.dst, f))]
            images = [Image.open(img) for img in i]
            t1 = time.perf_counter()
            log.info({ 'preprocess': { 'source': params.src, 'destination': params.dst, 'images': len(images), 'time': round(t1 - t0, 2) } })
            res = len(images)
        else:
            args.preprocess.process_dst = params.src
            i = [os.path.join(params.src, f) for f in os.listdir(params.src) if os.path.isfile(os.path.join(params.src, f)) and filetype.is_image(os.path.join(params.src, f))]
            images = [Image.open(img) for img in i]
            res = len(images)
    else:
        log.error({ 'preprocess error': { 'not a valid input': params.src } })
    if len(images) > 0:
        img = grid(images, labels = None, width = 2048, height = 2048, border = 8, square = True)
        logdir = os.path.abspath(os.path.join(cmdflags['embeddings_dir'], '../train/log'))
        fn = os.path.join(logdir, params.name + '-inputs.jpg')
        img.save(fn)
        log.info({ 'preprocess input grid': fn })
    return res


async def check(params):
    log.info({ 'checking server options' })
    global options # pylint: disable=global-statement
    options = await get('/sdapi/v1/options')

    options['training_xattention_optimizations'] = False
    options['training_image_repeats_per_epoch'] = 1

    log.debug({ 'check model': args.training_model })
    if len(args.training_model) > 0 and not options['sd_model_checkpoint'].startswith(args.training_model):
        models = await get('/sdapi/v1/sd-models')
        models = [obj["title"] for obj in models]
        found = [i for i in models if i.startswith(args.training_model)]
        if len(found) == 0:
            log.error({ 'model not found': args.training_model, 'available': models })
            exit()
        else:
            log.warning({ 'switching model': found[0] })
        options['sd_model_checkpoint'] = found[0]

    log.debug({ 'check embedding': params.name })
    global cmdflags # pylint: disable=global-statement
    cmdflags = await get('/sdapi/v1/cmd-flags')

    lst = os.path.join(cmdflags['embeddings_dir'])
    log.debug({ 'embeddings folder': lst })
    path = Path(cmdflags['embeddings_dir']).glob(f'{params.name}.pt*')
    matches = [f for f in path]
    for match in matches:
        if params.overwrite:
            log.info({ 'delete embedding': match.name })
            os.remove(os.path.join(cmdflags['embeddings_dir'], match.name))
        else:
            log.error({ 'embedding exists': match.name })
            await close()
            exit()
    logdir = os.path.abspath(os.path.join(cmdflags['embeddings_dir'], '../train/log', params.name))
    f = os.path.join(logdir, 'train.csv')
    if os.path.isfile(f):
        if params.overwrite:
            log.info({ 'delete training log': f })
            os.remove(os.path.join(logdir, 'train.csv'))
        else:
            log.warning({ 'training log exists': f })
    f = os.path.join(logdir, '..', params.name, '.png')
    if os.path.isfile(f):
        if params.overwrite:
            log.info({ 'delete training graph': f })
            os.remove(f)

    log.debug({ 'options': 'update' })
    await post('/sdapi/v1/options', options)
    return


async def create(params):
    log.debug({ 'create start' })
    if not os.path.isdir(args.preprocess.process_dst):
        log.error({ 'train source not found': args.preprocess.process_dst })
        exit()
    if params.vectors == -1: # dynamically determine number of vectors depending on number of input images
        if len(images) <= 20:
            vectors = 2
        elif len(images) <= 100:
            vectors = 4
        else:
            vectors = 6
    if os.path.exists(params.name) and os.path.isfile(params.name):
        log.info({ 'deleting existing embedding': { 'name': params.name } })
        os.remove(params.name)
    args.create_embedding.name = params.name
    words = params.init.split(',')
    if len(words) > vectors:
        params.init = ','.join(words[:vectors])
        log.warning({ 'create embedding init words cut': params.init })
    args.create_embedding.init_text = params.init
    args.create_embedding.num_vectors_per_token = vectors
    log.debug({ 'create args': args.create_embedding })
    res = await post('/sdapi/v1/create/embedding', args.create_embedding)
    if 'info' in res:
        log.info({ 'create embedding': { 'name': params.name, 'init': params.init, 'vectors': vectors, 'message': res.info } })
    else:
        log.error({ 'create failed:', res })
        return None
    log.debug({ 'create end' })
    return params.name


async def train(params):
    log.debug({ 'train start' })
    args.train_embedding.embedding_name = params.name
    args.train_embedding.data_root = args.preprocess.process_dst
    imgs = [f for f in os.listdir(args.preprocess.process_dst) if os.path.isfile(os.path.join(args.preprocess.process_dst, f)) and filetype.is_image(os.path.join(args.preprocess.process_dst, f))]
    if len(imgs) == 0:
        log.error({ 'train no input images in folder': args.preprocess.process_dst })
        return
    if params.grad == -1:
        grad = (len(imgs) // args.train_embedding.batch_size)
        args.train_embedding.gradient_step = max(grad, 30)
        log.info({ 'dynamic gradient step': args.train_embedding.gradient_step })
        if params.steps == -1:
            args.train_embedding.steps = 5000 // args.train_embedding.gradient_step
            log.info({ 'dynamic steps': args.train_embedding.steps })
    log.info({ 'train embedding': {
        'name': params.name,
        'source': args.preprocess.process_dst,
        'images': len(imgs),
        'steps': args.train_embedding.steps,
        'batch': args.train_embedding.batch_size,
        'gradient-step': args.train_embedding.gradient_step,
        'sampling': args.train_embedding.latent_sampling_method,
        'epoch-size': args.train_embedding.batch_size * args.train_embedding.gradient_step }
    })
    log.info({ 'learn-rate': args.train_embedding.learn_rate })
    log.debug({ 'train args': args.train_embedding })
    t0 = time.time()
    res = await post('/sdapi/v1/train/embedding', args.train_embedding)
    t1 = time.time()
    log.info({ 'train embedding finished': round(t1 - t0) })
    log.debug({ 'train end': res.info })
    return


async def pipeline(params):
    log.debug({ 'pipeline start' })

    # interrupt
    await interrupt()

    # preprocess
    num = await preprocess(params)
    if num == 0:
        log.warning({ 'preprocess': 'no resulting images'})
        return

    # create embedding
    name = await create(params)
    if not params.name in name:
        log.error({ 'create embedding failed': name })
        return

    # train embedding
    await train(params)

    await plotloss(params)

    log.debug({ 'pipeline end' })
    return


async def monitor(params):
    step = 0
    t0 = time.perf_counter()
    t1 = time.perf_counter()
    finished = 0
    while True:
        await asyncio.sleep(10)
        res = await progress()
        if not 'state' in res:
            log.info({ 'monitor disconnected': res })
            break
        if (res.state.job_count == params.steps and res.state.job_no >= res.state.job_count) or (res.eta_relative < 0) or (res.interrupted) or (res.state.job_count == 0): # need exit case if interrupted or failed
            if res.interrupted:
                log.info({ 'monitor interrupted': { 'embedding': params.name } })
                break # exit for monitor job
            else:
                finished += 1
            if finished >= 2: # do it more than once since preprocessing job can finish just in time for monitor to finish
                log.info({ 'monitor finished': { 'embedding': params.name } })
                break
        else:
            if res.state.job_no == 0:
                step = 0
                t0 = time.perf_counter()
                t1 = time.perf_counter()
            try:
                if 'Loss:' in res.textinfo:
                    text = res.textinfo.split('<br/>')[0].split()
                    loss = float(text[-1])
                else:
                    loss = -1
            except:
                loss = -1
            if math.isnan(loss):
                log.error({ 'monitor': { 'progress': round(100 * res.progress), 'embedding': params.name, 'eta': round(res.eta_relative), 'step': res.state.job_no, 'steps': res.state.job_count, 'loss': 'nan' } })
                await interrupt()
            else:
                elapsed = t1 - t0
                log.info({ 'monitor': {
                    'job': res.state.job,
                    'progress': round(100 * res.progress),
                    'embedding': params.name,
                    'epoch': (1 + res.state.job_no // len(images)) if len(images) > 0 else 'n/a',
                    'step': res.state.job_no,
                    'steps': res.state.job_count,
                    'loss': loss if loss > -1 else 'n/a',
                    'total': round(1.0 * elapsed * res.state.job_count / res.state.job_no) if res.state.job_no > 0 and t1 != t0 else 'n/a',
                    'elapsed': round(elapsed),
                    'remaining': round(res.eta_relative),
                    'it/s': round((res.state.job_no - step) / (time.perf_counter() - t1), 2) }
                })
                if step % 10 == 0:
                    await plotloss(params)
                step = res.state.job_no
            t1 = time.perf_counter()
    return


async def main():
    parser = argparse.ArgumentParser(description="sd train pipeline")
    parser.add_argument("--config", type = str, default = 'train.json', required = False, help = "configuration file, default: %(default)s")
    parser.add_argument("--name", type = str, required = True, help = "embedding name, set to auto to use src folder name")
    parser.add_argument("--src", type = str, required = True, help = "source image folder or movie file")
    parser.add_argument("--init", type = str, default = "person", required = False, help = "initialization class, default: %(default)s")
    parser.add_argument("--dst", type = str, default = "/tmp", required = False, help = "destination image folder for processed images, default: %(default)s")
    parser.add_argument("--steps", type = int, default = -1, required = False, help = "training steps, default: %(default)s")
    parser.add_argument("--vectors", type = int, default = -1, required = False, help = "number of vectors per token, default: dynamic based on number of input images")
    parser.add_argument("--batch", type = int, default = 1, required = False, help = "batch size, default: %(default)s")
    parser.add_argument("--rate", type = str, default = "", required = False, help = "learning rate, default: dynamic")
    parser.add_argument("--rstart", type = float, default = 0.01, required = False, help = "starting learn rate if using dynamic rate, default: %(default)s")
    parser.add_argument("--rend", type = float, default = 0.0001, required = False, help = "ending learn rate if using dynamic rate, default: %(default)s")
    parser.add_argument("--rdescend", type = float, default = 2, required = False, help = "learn rate descend power when using dynamic rate, default: %(default)s")
    parser.add_argument("--grad", type = int, default = -1, required = False, help = "accumulate gradient over n images, default: : %(default)s")
    parser.add_argument("--type", type = str, default = 'subject', required = False, help = "training type: subject/style/unknown, default: %(default)s")
    parser.add_argument('--overwrite', default = False, action='store_true', help = "overwrite existing embedding, default: %(default)s")
    parser.add_argument("--vstart", type = float, default = 0, required = False, help = "if processing video skip first n seconds, default: %(default)s")
    parser.add_argument("--vend", type = float, default = 0, required = False, help = "if processing video skip last n seconds, default: %(default)s")
    parser.add_argument('--skipcaption', default = False, action='store_true', help = "do not auto-generate captions, default: %(default)s")
    parser.add_argument('--preprocess', type = str, choices=['builtin', 'custom', 'none'], default = 'custom', help = "preprocessing type, default: %(default)s")
    parser.add_argument('--nocleanup', default = False, action='store_true', help = "skip cleanup after completion, default: %(default)s")
    parser.add_argument('--debug', default = False, action='store_true', help = "print extra debug information, default: %(default)s")
    params = parser.parse_args()
    if params.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': params.__dict__ })
    home = Path(sys.argv[0]).parent
    global args # pylint: disable=global-statement
    if os.path.isfile(params.config):
        try:
            with open(params.config, 'r', encoding='utf-8') as f:
                data = json.load(f)
                args = Map(data) # pylint: disable=redefined-outer-name
                log.debug({ 'config': args })
        except Exception as e:
            log.error({ 'config error': params.config, 'exception': e })
            exit()
    elif os.path.isfile(os.path.join(home, params.config)):
        try:
            with open(os.path.join(home, params.config), 'r', encoding='utf-8') as f:
                data = json.load(f)
                args = Map(data) # pylint: disable=redefined-outer-name
                log.debug({ 'config': args })
        except Exception as e:
            log.error({ 'config error': params.config, 'exception': e })
            exit()
    else:
        log.error({ 'config file not found': params.config})
        exit()
    if params.vstart > 0:
        args.extract_video.vstart = params.vstart
    if params.vend > 0:
        args.extract_video.vend = params.vend
    if params.steps > -1:
        args.train_embedding.steps = params.steps
    if params.batch > -1:
        args.train_embedding.batch_size = params.batch
    if params.rate != '':
        args.train_embedding.learn_rate = params.rate
    if params.grad > -1:
        args.train_embedding.gradient_step = params.grad
    epoch_size = args.train_embedding.batch_size * args.train_embedding.gradient_step
    if args.train_embedding.create_image_every == -1:
        args.train_embedding.create_image_every = epoch_size
    if args.train_embedding.save_embedding_every == -1:
        args.train_embedding.save_embedding_every = epoch_size
    if args.train_embedding.learn_rate == -1:
        loss_args = {
            "steps": args.train_embedding.steps,
            "step": epoch_size,
            "loss_start": params.rstart,
            "loss_end": params.rend,
            "loss_type": 'power',
            "power": params.rdescend
        }
        args.train_embedding.learn_rate = gen_loss_rate_str(**loss_args)
        log.debug({ 'learning rate': args.train_embedding.learn_rate, 'params': loss_args })
    if params.type == 'subject':
        if params.skipcaption:
            args.train_embedding.template_filename = 'subject.txt'
            args.preprocess.process_caption = False
        else:
            args.train_embedding.template_filename = 'subject_filewords.txt'
    elif params.type == 'style':
        if params.skipcaption:
            args.train_embedding.template_filename = 'style.txt'
            args.preprocess.process_caption = False
        else:
            args.train_embedding.template_filename = 'style_filewords.txt'
    else:
        if params.skipcaption:
            args.train_embedding.template_filename = 'unknown.txt'
            args.preprocess.process_caption = False
        else:
            args.train_embedding.template_filename = 'unknown_filewords.txt'
    if params.name == 'auto':
        params.name = PurePath(params.src).name
        log.info({ 'training name': params.name })
    if params.dst == "/tmp":
        params.dst = os.path.join("/tmp/train", params.name)
    log.debug({ 'args': params.__dict__ })
    params.src = os.path.abspath(params.src)
    params.dst = os.path.abspath(params.dst)
    try:
        await session()
        await check(params)
        a = asyncio.create_task(pipeline(params))
        b = asyncio.create_task(monitor(params))
        await asyncio.gather(a, b) # wait for both pipeline and monitor to finish
    except Exception as e:
        log.error({ 'exception': e })
    finally:
        await cleanup(params)
        await close()
    return

if __name__ == "__main__": # create & train test embedding when used from cli
    log.info({ 'train script' })
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning({ 'interrupted': 'keyboard request' })
        # asyncio.run(interrupt())
