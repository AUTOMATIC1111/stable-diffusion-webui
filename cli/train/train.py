#!/bin/env python

# system imports
import os
import re
import gc
import sys
import json
import shutil
import pathlib
import asyncio
import tempfile
import argparse

# 3rd party imports
import filetype
from tqdm.rich import tqdm

# local imports
import util
import sdapi
import process
import latents
import options

# console handler
from rich import print # pylint: disable=redefined-builtin
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console
console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
pretty_install(console=console)
import torch, accelerate, diffusers, requests, urllib3, http
traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[torch,accelerate,diffusers,asyncio,http,urllib3,requests])

# lora imports
lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lora'))
sys.path.append(lora_path)
lycoris_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lycoris'))
sys.path.append(lycoris_path)
import train_network


# globals
args = None
valid_steps = ['original', 'face', 'body', 'blur', 'range', 'upscale', 'restore', 'interrogate', 'resize', 'square', 'segment']

# methods

def mem_stats():
    gc.collect()
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    mem = util.get_memory()
    peak = { 'active': mem['gpu-active']['peak'], 'allocated': mem['gpu-allocated']['peak'], 'reserved': mem['gpu-reserved']['peak'] }
    console.log(f"memory cpu: {mem.ram} gpu current: {mem.gpu} gpu peak: {peak}")


def parse_args():
    global args
    parser = argparse.ArgumentParser(description = 'train lora')
    # basic section
    parser.add_argument('--output', '--name', type=str, default=None, required=True, help='output filename')
    parser.add_argument('--type', type=str, choices=['embedding', 'lora', 'lycoris', 'dreambooth'], default=None, required=True, help='training type')
    parser.add_argument('--tag', type=str, default='person', required=False, help='primary tag, default: %(default)s')
    parser.add_argument('--process', type=str, default='original,interrogate,resize,square', required=False, help=f'list of possible processing steps: {valid_steps}, default: %(default)s')
    parser.add_argument('--dir', type=str, default='', required=False, help='where to store processed images, default is system temp/train')
    parser.add_argument('--input', '--dataset', type=str, default=None, required=True, help='input folder with training images')
    parser.add_argument('--overwrite', default = False, action='store_true', help = "overwrite existing training, default: %(default)s")

    # global params
    parser.add_argument('--gradient', type=int, default=1, required=False, help='gradient accumulation steps, default: %(default)s')
    parser.add_argument('--steps', type=int, default=2500, required=False, help='training steps, default: %(default)s')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size, default: %(default)s')
    parser.add_argument('--lr', type=float, default=1e-04, required=False, help='model learning rate, default: %(default)s')
    parser.add_argument('--dim', '--vectors', type=int, default=40, required=False, help='network dimension, default: %(default)s')

    # lora params
    parser.add_argument('--repeats', type=int, default=10, required=False, help='number of repeats per image, default: %(default)s')
    parser.add_argument('--alpha', type=float, default=0, required=False, help='alpha for weights scaling, default: half of dim')

    args = parser.parse_args()


def prepare_server():
    try:
        server_status = util.Map(sdapi.progress())
        server_state = server_status['state']
    except:
        console.log('server error:', server_status)
        exit(1)
    if server_state['job_count'] > 0:
        console.log('server not idle:', server_state)
        exit(1)

    server_options = util.Map(sdapi.options())
    server_options.options.save_training_settings_to_txt = False
    server_options.options.training_enable_tensorboard = False
    server_options.options.training_tensorboard_save_images = False
    server_options.options.pin_memory = True
    server_options.options.save_optimizer_state = False
    server_options.options.training_image_repeats_per_epoch = args.repeats
    server_options.options.training_write_csv_every = 0
    sdapi.postsync('/sdapi/v1/options', server_options.options)
    console.log(f'updated server options')


def verify_args():
    global args
    server_options = util.Map(sdapi.options())
    args.model = server_options.options['sd_model_checkpoint'].split(' [')[0]
    args.lora_dir = server_options.flags['lora_dir']
    if not os.path.isabs(args.model) and not os.path.exists(args.model):
        args.model = os.path.abspath(os.path.join(args.lora_dir, os.pardir, 'Stable-diffusion', args.model))

    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        console.log('cannot find model:', args.model)
        exit(1)
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        console.log('cannot find training folder:', args.input)
        exit(1)
    if not os.path.exists(args.lora_dir) or not os.path.isdir(args.lora_dir):
        console.log('cannot find lora folder:', args.dir)
        exit(1)
    if args.dir != '':
        args.process_dir = args.dir
    else:
        args.process_dir = os.path.join(tempfile.gettempdir(), 'train', args.output)
    console.log(f'args: {vars(args)}')


async def training_loop():
    async def async_train():
        res = await sdapi.post('/sdapi/v1/train/embedding', options.embedding)
        console.log(f'train embedding result: {res}')

    async def async_monitor():
        await asyncio.sleep(3)
        res = util.Map(sdapi.progress())
        with tqdm(desc='train embedding', total=res.state.job_count) as pbar:
            while res.state.job_no < res.state.job_count and not res.state.interrupted and not res.state.skipped:
                await asyncio.sleep(2)
                prev_job = res.state.job_no
                res = util.Map(sdapi.progress())
                loss = re.search(r"Loss: (.*?)(?=\<)", res.textinfo)
                if loss:
                    pbar.set_postfix({ 'loss': loss.group(0) })
                    pbar.update(res.state.job_no - prev_job)

    a = asyncio.create_task(async_train())
    b = asyncio.create_task(async_monitor())
    await asyncio.gather(a, b) # wait for both pipeline and monitor to finish


def train_embedding():
    console.log(f'{args.type} options: {options.embedding}')
    create_options = util.Map({
        "name": args.output,
        "num_vectors_per_token": args.dim,
        "overwrite_old": False,
        "init_text": args.tag,
    })
    server_options = util.Map(sdapi.options())
    fn = os.path.join(server_options.flags.embeddings_dir, args.output) + '.pt'
    if os.path.exists(fn) and args.overwrite:
        console.log(f'delete existing embedding {fn}')
        os.remove(fn)
    else:
        console.log(f'embedding exists {fn}')
        return
    console.log(f'create embedding {create_options}')
    res = sdapi.postsync('/sdapi/v1/create/embedding', create_options)
    if 'info' in res and 'error' in res['info']: # formatted error
        console.log(res.info)
    elif 'info' in res: # no error
        asyncio.run(training_loop())
    else: # unknown error
        console.log(f'create embedding error {res}')


def train_lora():
    fn = os.path.join(args.lora_dir, args.output)
    for ext in ['.ckpt', '.pt', '.safetensors']:
        if os.path.exists(fn + ext):
            if args.overwrite:
                console.log(f'delete existing lora: {fn + ext}')
                os.remove(fn + ext)
            else:
                console.log(f'lora exists: {fn + ext}')
                return
    console.log(f'{args.type} options: {options.lora}')
    train_network.train(options.lora)


def prepare_options():
    # lora specific
    options.lora.pretrained_model_name_or_path = args.model
    options.lora.output_dir = args.lora_dir
    options.lora.output_name = args.output
    options.lora.max_train_steps = args.steps
    options.lora.network_dim = args.dim
    options.lora.network_alpha = args.dim // 2 if args.alpha == 0 else args.alpha
    options.lora.gradient_accumulation_steps = args.gradient
    options.lora.learning_rate = args.lr
    options.lora.train_batch_size = args.batch
    options.lora.network_alpha = args.dim // 2 if args.alpha == 0 else args.alpha
    options.lora.train_data_dir = args.process_dir
    if args.type == 'lycoris':
        console.log('train using lycoris network')
        options.lora.network_module = 'lycoris.kohya'
        options.lora.in_json = os.path.join(args.process_dir, args.output + '.json')
    if args.type == 'dreambooth':
        console.log('train using dreambooth style training')
        options.lora.in_json = None
    if args.type == 'lora':
        console.log('train using lora style training')
        options.lora.in_json = os.path.join(args.process_dir, args.output + '.json')
    if args.type == 'embedding':
        console.log('train embedding')
        options.lora.in_json = None
        pass
    # embedding specific
    options.embedding.embedding_name = args.output
    options.embedding.learn_rate = str(args.lr)
    options.embedding.batch_size = args.batch
    options.embedding.steps = args.steps
    options.embedding.data_root = args.process_dir
    options.embedding.log_directory = os.path.join(args.process_dir, 'log')
    options.embedding.gradient_step = args.gradient


def process_inputs():
    pathlib.Path(args.process_dir).mkdir(parents=True, exist_ok=True)
    processing_options = args.process.split(',') if isinstance(args.process, str) else args.process
    processing_options = [opt.strip() for opt in re.split(',| ', args.process)]
    console.log(f'processing steps: {processing_options}')
    for step in processing_options:
        if step not in valid_steps:
            console.log(f'invalid processing step: {[step]}')
            exit(1)
    for root, _sub_dirs, folder in os.walk(args.input):
        files = [os.path.join(root, f) for f in folder if filetype.is_image(os.path.join(root, f))]
    console.log(f'processing input images: {len(files)}')
    if os.path.exists(args.process_dir):
        console.log('removing existing processed folder:', args.process_dir)
        shutil.rmtree(args.process_dir, ignore_errors=True)
    steps = [step for step in processing_options if step in ['face', 'body', 'original']]
    process.reset()
    metadata = {}
    for step in steps:
        if step == 'face':
            opts = [step for step in processing_options if step not in ['body', 'original']]
        if step == 'body':
            opts = [step for step in processing_options if step not in ['face', 'original', 'upscale', 'restore']] # body does not perform upscale or restore
        if step == 'original':
            opts = [step for step in processing_options if step not in ['face', 'body', 'upscale', 'restore', 'blur', 'range', 'segment']] # original does not perform most steps
        console.log(f'processing current step: {opts}')
        tag = step
        if tag == 'original' and args.tag is not None:
            concept = args.tag.split(',')[0].strip()
        else:
            concept = step
        if args.type in ['lora', 'lycoris', 'dreambooth']:
            dir = os.path.join(args.process_dir, str(args.repeats) + '_' + concept) # separate concepts per folder
        if args.type in ['embedding']:
            dir = os.path.join(args.process_dir) # everything into same folder
        console.log('processing concept:', concept)
        console.log('processing output folder:', dir)
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        results = {}
        for f in files:
            res = process.file(filename = f, folder = dir, tag = args.tag, requested = opts)
            if res.image: # valid result
                results[res.type] = results.get(res.type, 0) + 1
                results['total'] = results.get('total', 0) + 1
                rel_path = res.basename.replace(os.path.commonpath([res.basename, args.process_dir]), '')
                if rel_path.startswith(os.path.sep): rel_path = rel_path[1:]
                metadata[rel_path] = { 'caption': res.caption, 'tags': ','.join(res.tags) }
                if options.lora.in_json is None:
                    with open(res.output.replace(options.process.format, '.txt'), "w") as outfile:
                        outfile.write(res.caption)
            console.log(f"processing {'saved' if res.image is not None else 'skipped'}: {f} => {res.output} {res.ops} {res.message}")
    dirs = [os.path.join(args.process_dir, dir) for dir in os.listdir(args.process_dir) if os.path.isdir(os.path.join(args.process_dir, dir))]
    console.log(f'input datasets {dirs}')
    if options.lora.in_json is not None:
        with open(options.lora.in_json, "w") as outfile: # write json at the end only
            outfile.write(json.dumps(metadata, indent=2))
        for dir in dirs: # create latents
            latents.create_vae_latents(util.Map({ 'input': dir, 'json': options.lora.in_json }))
            latents.unload_vae()
    r = { 'inputs': len(files), 'outputs': results, 'metadata': options.lora.in_json }
    console.log(f'processing steps result: {r}')
    if args.gradient < 0:
        console.log(f"setting gradient accumulation to number of images: {results['total']}")
        options.lora.gradient_accumulation_steps = results['total']
        options.embedding.gradient_step = results['total']
    process.unload()


if __name__ == '__main__':
    console.log('train script for stable diffusion')
    parse_args()
    prepare_server()
    verify_args()
    prepare_options()
    mem_stats()
    process_inputs()
    mem_stats()
    try:
        if args.type == 'embedding':
            train_embedding()
        if args.type == 'lora' or args.type == 'lycoris' or args.type == 'dreambooth':
            train_lora()
    except KeyboardInterrupt as e:
        console.log('interrupt requested')
        sdapi.interrupt()
    mem_stats()
    console.log('done')
