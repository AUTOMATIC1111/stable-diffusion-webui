#!/bin/env python
"""
sd api txt2img benchmark
"""
import asyncio
import base64
import io
import json
import time
from PIL import Image
import sdapi
from util import Map, log


options = Map({
    'restore_faces': False,
    'prompt': 'photo of two dice on a table',
    'negative_prompt': 'foggy, blurry',
    'steps': 20,
    'batch_size': 1,
    'n_iter': 1,
    'seed': -1,
    'sampler_name': 'Euler a',
    'cfg_scale': 0,
    'width': 512,
    'height': 512
})


# batch = [1, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
batch = [1, 1, 2, 4, 8, 12, 16]
oom = 0


async def txt2img():
    t0 = time.perf_counter()
    data = {}
    try:
        data = await sdapi.post('/sdapi/v1/txt2img', options)
    except:
        return -1
    if 'error' in data:
        return -1
    if 'info' in data:
        info = Map(json.loads(data['info']))
    else:
        return 0
    log.debug({ 'info': info })
    for i in range(len(data['images'])):
        data['images'][i] = Image.open(io.BytesIO(base64.b64decode(data['images'][i].split(',',1)[0])))
        log.debug({ 'image': data['images'][i].size })
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
    log.info({ 'benchmark': { 'batch-sizes': batch } })
    sdapi.quiet = True
    await sdapi.session()
    await sdapi.interrupt()
    opts = await sdapi.get('/sdapi/v1/options')
    opts = Map(opts)
    log.info({ 'options': {
        'resolution': [options.width, options.height],
        'model': opts.sd_model_checkpoint,
        'vae': opts.sd_vae,
        'hypernetwork': opts.sd_hypernetwork,
        'sampler': options.sampler_name,
        'clip-stop': opts.CLIP_stop_at_last_layers,
        'preview': opts.show_progress_every_n_steps
    } })
    cpu, gpu = memstats()
    log.info({ 'system': { 'cpu': cpu, 'gpu': gpu }})
    for i in range(len(batch)):
        if oom > 0:
            continue
        options['batch_size'] = batch[i]
        ts = await txt2img()
        if ts > 0:
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.warning({ 'interrupted': 'keyboard request' })
        sdapi.interruptsync()
