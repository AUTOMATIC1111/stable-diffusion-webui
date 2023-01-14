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
import pathlib
import time
import json
from pathlib import Path

import filetype
from ffmpeg import extract
from sdapi import close, get, interrupt, post, progress, session
from util import Map, log
from losschart import plot
from PIL import Image

images = []
use_pbar = False

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
    log.info({ 'cleanup deleting preprocessed images': params.dst })
    for f in Path(params.dst).glob('*.png'):
        f.unlink()
    for f in Path(params.dst).glob('*.txt'):
        f.unlink()


async def preprocess(params):
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
    if os.path.isdir(params.src):
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
    elif os.path.isfile(params.src):
        if not filetype.is_video(params.src):
            kind = filetype.guess(params.src)
            log.error({ 'preprocess error': { 'not a valid movie file': params.src, 'guess': kind } })
        else:
            extract_dst = os.path.join(params.dst, 'extract')
            log.debug({ 'preprocess args': args.extract_video })
            images = extract(params.src, extract_dst, rate = args.extract_video.rate, fps = args.extract_video.fps, start = args.extract_video.skipstart, end = args.extract_video.skipend) # extract keyframes from movie
            if images > 0:
                params.src = extract_dst
                processed_count = await preprocess(params) # call again but now with keyframes
                return processed_count
            else:
                log.error({ 'preprocess video extract': 'no images' })
                return 0
    else:
        log.error({ 'preprocess error': { 'not a valid input': params.src } })
        return 0


async def check(params):
    log.debug({ 'setting options' })
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
    args.create_embedding.init_text = params.init
    args.create_embedding.num_vectors_per_token = vectors
    log.debug({ 'create args': args.create_embedding })
    res = await post('/sdapi/v1/create/embedding', args.create_embedding)
    log.info({ 'create embedding': { 'name': params.name, 'init': params.init, 'vectors': vectors, 'message': res.info } })
    log.debug({ 'create end' })
    return params.name


async def train(params):
    log.debug({ 'train start' })
    args.train_embedding.embedding_name = params.name
    args.train_embedding.data_root = args.preprocess.process_dst
    if params.accumulation > -1:
        args.train_embedding.gradient_step = params.accumulation
    log.info({ 'train': {
        'name': params.name,
        'source': args.preprocess.process_dst,
        'steps': args.train_embedding.steps,
        'batch': args.train_embedding.batch_size,
        'accumulation': args.train_embedding.gradient_step,
        'learning-rate': args.train_embedding.learn_rate }
    })
    log.debug({ 'train args': args.train_embedding })
    res = await post('/sdapi/v1/train/embedding', args.train_embedding)
    log.debug({ 'train end': res.info })
    return


async def pipeline(params):
    log.debug({ 'pipeline start' })

    # interrupt
    await interrupt()

    # preproceess
    if not params.skippreprocess:
        num = await preprocess(params)
        if num == 0:
            log.warning({ 'preprocess': 'no resulting images'})
            return
    else:
        args.preprocess.process_dst = params.src

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
    parser.add_argument("--vectors", type = int, default = -1, required = False, help = "number of vectors per token, default: dynamic")
    parser.add_argument("--batch", type = int, default = -1, required = False, help = "batch size, default: %(default)s")
    parser.add_argument("--rate", type = str, default = "", required = False, help = "learning rate, default: dynamic")
    parser.add_argument("--accumulation", type = int, default = 10, required = False, help = "accumulate gradient over n steps, default: dynamic")
    parser.add_argument("--type", choices = ['subject', 'style'], default = 'subject', required = False, help = "subject or style, default: %(default)s")
    parser.add_argument('--overwrite', default = False, action='store_true', help = "overwrite existing embedding, default: %(default)s")
    parser.add_argument('--skipcaption', default = False, action='store_true', help = "do not auto-generate captions, default: %(default)s")
    parser.add_argument('--skippreprocess', default = False, action='store_true', help = "skip preprocessing, default: %(default)s")
    parser.add_argument("--skipstart", type = float, default = -1, required = False, help = "if processing video skip first n seconds, default: %(default)s")
    parser.add_argument("--skipend", type = float, default = -1, required = False, help = "if processing video skip last n seconds, default: %(default)s")
    parser.add_argument('--nocleanup', default = False, action='store_true', help = "skip cleanup after completion, default: %(default)s")
    parser.add_argument('--debug', default = False, action='store_true', help = "print extra debug information, default: %(default)s")
    params = parser.parse_args()
    if params.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': params.__dict__ })
    home = pathlib.Path(sys.argv[0]).parent
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
    if params.skipstart > -1:
        args.extract_video.skipstart = params.skipstart
    if params.skipend > -1:
        args.extract_video.skipend = params.skipend
    if params.steps > -1:
        args.train_embedding.steps = params.steps
    if params.batch > -1:
        args.train_embedding.batch_size = params.batch
    if params.rate != '':
        args.train_embedding.learn_rate = params.rate
    if params.type == 'subject':
        if params.skipcaption:
            args.train_embedding.template_filename = 'subject.txt'
            args.preprocess.process_caption = False
        else:
            args.train_embedding.template_filename = 'subject_filewords.txt'
    if params.type == 'style':
        if params.skipcaption:
            args.train_embedding.template_filename = 'style.txt'
            args.preprocess.process_caption = False
        else:
            args.train_embedding.template_filename = 'style_filewords.txt'
    if params.name == 'auto':
        params.name = pathlib.PurePath(params.src).name
        log.info({ 'training name': params.name })
    if params.dst == "/tmp":
        params.dst = os.path.join("/tmp/train", params.name)
    log.info({ 'args': params.__dict__ })
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
    log.info({ 'train embedding' })
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning({ 'interrupted': 'keyboard request' })
        # asyncio.run(interrupt())
