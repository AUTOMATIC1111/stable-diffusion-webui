#!/usr/bin/env python
"""
sd api txt2img benchmark
"""
import os
import asyncio
import base64
import io
import json
import time
import argparse
from PIL import Image
import sdapi
from util import Map, log


oom = 0
args = None
options = None


async def txt2img():
    t0 = time.perf_counter()
    data = {}
    try:
        data = await sdapi.post('/sdapi/v1/txt2img', options)
    except Exception:
        return -1
    if 'error' in data:
        return -1
    if 'info' in data:
        info = Map(json.loads(data['info']))
    else:
        return 0
    log.debug({ 'info': info })
    if options['batch_size'] != len(data['images']):
        log.error({ 'requested': options['batch_size'], 'received': len(data['images']) })
    for i in range(len(data['images'])):
        data['images'][i] = Image.open(io.BytesIO(base64.b64decode(data['images'][i].split(',',1)[0])))
        if args.save:
            fn = os.path.join(args.save, f'benchmark-{i}-{len(data["images"])}.png')
            data["images"][i].save(fn)
            log.debug({ 'save': fn })
    log.debug({ "images": data["images"] })
    t1 = time.perf_counter()
    return t1 - t0


def memstats():
    mem = sdapi.getsync('/sdapi/v1/memory')
    cpu = mem.get('ram', 'unavailable')
    gpu = mem.get('cuda', 'unavailable')
    if 'active' in gpu:
        gpu['session'] = gpu.pop('active')
    if 'reserved' in gpu:
        gpu.pop('allocated')
        gpu.pop('reserved')
        gpu.pop('inactive')
    if 'events' in gpu:
        global oom # pylint: disable=global-statement
        oom = gpu['events']['oom']
        gpu.pop('events')
    return cpu, gpu


def gb(val: float):
    return round(val / 1024 / 1024 / 1024, 2)


async def main():
    sdapi.quiet = True
    await sdapi.session()
    await sdapi.interrupt()
    ver = await sdapi.get("/sdapi/v1/version")
    log.info({ 'version': ver})
    platform = await sdapi.get("/sdapi/v1/platform")
    log.info({ 'platform': platform })
    opts = await sdapi.get('/sdapi/v1/options')
    opts = Map(opts)
    log.info({ 'model': opts.sd_model_checkpoint })
    cpu, gpu = memstats()
    log.info({ 'system': { 'cpu': cpu, 'gpu': gpu }})
    batch = [1, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    batch = [b for b in batch if b <= args.maxbatch]
    log.info({"batch-sizes": batch})
    for i in range(len(batch)):
        if oom > 0:
            continue
        options['batch_size'] = batch[i]
        warmup = await txt2img()
        ts = await txt2img()
        if i == 0:
            ts += warmup
        if ts > 0.01: # cannot be faster than 10ms per run
            await asyncio.sleep(0)
            cpu, gpu = memstats()
            if i == 0:
                log.info({ 'warmup': round(ts, 2) })
            else:
                peak = gpu['session']['peak'] if 'session' in gpu else 0
                log.info({ 'batch': batch[i], 'its': round(options.steps / (ts / batch[i]), 2), 'img': round(ts / batch[i], 2), 'wall': round(ts, 2), 'peak': gb(peak), 'oom': oom > 0 })
        else:
            await asyncio.sleep(10)
            cpu, gpu = memstats()
            log.info({ 'batch': batch[i], 'result': 'error', 'gpu': gpu, 'oom': oom > 0 })
            break
    if oom > 0:
        log.info({ 'benchmark': 'ended with oom so you should probably restart your automatic server now' })
    await sdapi.close()


if __name__ == '__main__':
    log.info({ 'run-benchmark' })
    parser = argparse.ArgumentParser(description = 'run-benchmark')
    parser.add_argument("--steps", type=int, default=50, required=False, help="steps")
    parser.add_argument("--sampler", type=str, default='Euler a', required=False, help="max batch size")
    parser.add_argument("--prompt", type=str, default='photo of two dice on a table', required=False, help="prompt")
    parser.add_argument("--negative", type=str, default='foggy, blurry', required=False, help="prompt")
    parser.add_argument("--maxbatch", type=int, default=16, required=False, help="max batch size")
    parser.add_argument("--width", type=int, default=512, required=False, help="width")
    parser.add_argument("--height", type=int, default=512, required=False, help="height")
    parser.add_argument('--debug', default = False, action='store_true', help = 'debug logging')
    parser.add_argument('--taesd', default = False, action='store_true', help = 'use taesd as vae')
    parser.add_argument("--save", type=str, default='', required=False, help="save images to folder")
    args = parser.parse_args()
    if args.debug:
        log.setLevel('DEBUG')
    options = Map(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative,
            "steps": args.steps,
            "sampler_name": args.sampler,
            "width": args.width,
            "height": args.height,
            "full_quality": not args.taesd,
            "cfg_scale": 0,
            "batch_size": 1,
            "n_iter": 1,
            "seed": -1,
        }
    )
    log.info({"options": options})
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning({ 'interrupted': 'keyboard request' })
        sdapi.interruptsync()
