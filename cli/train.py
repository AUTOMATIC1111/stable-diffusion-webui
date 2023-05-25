#!/usr/bin/env python

# system imports
import os
import re
import gc
import sys
import json
import time
import shutil
import pathlib
import asyncio
import logging
import tempfile
import argparse
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

# 3rd party imports
import filetype
import torch
from tqdm.rich import tqdm

# local imports
import util
import sdapi
import options
import process
import latents


# globals
args = None
log = logging.getLogger('train')
valid_steps = ['original', 'face', 'body', 'blur', 'range', 'upscale', 'restore', 'interrogate', 'resize', 'square', 'segment']
log_file = os.path.join(os.path.dirname(__file__), 'train.log')

# methods

def setup_logging(clean=False):
    try:
        if clean and os.path.isfile(log_file):
            os.remove(log_file)
        time.sleep(0.1) # prevent race condition
    except:
        pass
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
        "traceback.border": "black",
        "traceback.border.syntax_error": "black",
        "inspect.value.border": "black",
    }))
    # logging.getLogger("urllib3").setLevel(logging.ERROR)
    # logging.getLogger("httpx").setLevel(logging.ERROR)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s', filename=log_file, filemode='a', encoding='utf-8', force=True)
    log.setLevel(logging.DEBUG) # log to file is always at level debug for facility `sd`
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[])
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=level, console=console)
    rh.set_name(level)
    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])
    log.addHandler(rh)


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
    log.debug(f"memory cpu: {mem.ram} gpu current: {mem.gpu} gpu peak: {peak}")


def parse_args():
    global args # pylint: disable=global-statement
    parser = argparse.ArgumentParser(description = 'SD.Next Train')

    group_main = parser.add_argument_group('Main')
    group_main.add_argument('--type', type=str, choices=['embedding', 'ti', 'lora', 'lyco', 'dreambooth', 'hypernetwork'], default=None, required=True, help='training type')
    group_main.add_argument('--model', type=str, default='', required=False, help='base model to use for training, default: current loaded model')
    group_main.add_argument('--name', type=str, default=None, required=True, help='output filename')
    group_main.add_argument('--tag', type=str, default='person', required=False, help='primary tags, default: %(default)s')

    group_data = parser.add_argument_group('Dataset')
    group_data.add_argument('--input', type=str, default=None, required=True, help='input folder with training images')
    group_data.add_argument('--output', type=str, default='', required=False, help='where to store processed images, default is system temp/train')
    group_data.add_argument('--process', type=str, default='original,interrogate,resize,square', required=False, help=f'list of possible processing steps: {valid_steps}, default: %(default)s')

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--gradient', type=int, default=1, required=False, help='gradient accumulation steps, default: %(default)s')
    group_train.add_argument('--steps', type=int, default=2500, required=False, help='training steps, default: %(default)s')
    group_train.add_argument('--batch', type=int, default=1, required=False, help='batch size, default: %(default)s')
    group_train.add_argument('--lr', type=float, default=1e-04, required=False, help='model learning rate, default: %(default)s')
    group_train.add_argument('--dim', type=int, default=32, required=False, help='network dimension or number of vectors, default: %(default)s')

    # lora params
    group_train.add_argument('--repeats', type=int, default=10, required=False, help='number of repeats per image, default: %(default)s')
    group_train.add_argument('--alpha', type=float, default=0, required=False, help='lora/lyco alpha for weights scaling, default: dim/2')
    group_train.add_argument('--algo', type=str, default=None, choices=['locon', 'loha', 'lokr', 'ia3'], required=False, help='alternative lyco algoritm, default: %(default)s')
    group_train.add_argument('--args', type=str, default=None, required=False, help='lora/lyco additional network arguments, default: %(default)s')

    group_other = parser.add_argument_group('Other')
    group_other.add_argument('--overwrite', default = False, action='store_true', help = "overwrite existing training, default: %(default)s")
    group_other.add_argument('--debug', default = False, action='store_true', help = "enable debug level logging, default: %(default)s")

    args = parser.parse_args()


def prepare_server():
    try:
        server_status = util.Map(sdapi.progress())
        server_state = server_status['state']
    except:
        log.error(f'server error: {server_status}')
        exit(1)
    if server_state['job_count'] > 0:
        log.error(f'server not idle: {server_state}')
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
    log.info('updated server options')


def verify_args():
    server_options = util.Map(sdapi.options())
    if args.model != '':
        if not os.path.isfile(args.model):
            log.error(f'cannot find loaded model: {args.model}')
            exit(1)
        server_options.options.sd_model_checkpoint = args.model
        sdapi.postsync('/sdapi/v1/options', server_options.options)
    else:
        args.model = server_options.options.sd_model_checkpoint.split(' [')[0]
    args.lora_dir = server_options.options.lora_dir
    args.lyco_dir = server_options.options.lyco_dir
    args.ckpt_dir = server_options.options.ckpt_dir
    args.embeddings_dir = server_options.options.embeddings_dir
    if not os.path.isfile(args.model):
        attempt = os.path.abspath(os.path.join(args.ckpt_dir, args.model))
        args.model = attempt if os.path.isfile(attempt) else args.model
    if not os.path.isfile(args.model):
        attempt = os.path.abspath(os.path.join(args.ckpt_dir, '..', args.model))
        args.model = attempt if os.path.isfile(attempt) else args.model
    if not os.path.isfile(args.model):
        log.error(f'cannot find loaded model: {args.model}')
        exit(1)
    if not os.path.exists(args.ckpt_dir) or not os.path.isdir(args.ckpt_dir):
        log.error(f'cannot find models folder: {args.ckpt_dir}')
        exit(1)
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        log.error(f'cannot find training folder: {args.input}')
        exit(1)
    if not os.path.exists(args.lora_dir) or not os.path.isdir(args.lora_dir):
        log.error(f'cannot find lora folder: {args.lora_dir}')
        exit(1)
    if not os.path.exists(args.lyco_dir) or not os.path.isdir(args.lyco_dir):
        log.error(f'cannot find lyco folder: {args.lyco_dir}')
        exit(1)
    if args.output != '':
        args.process_dir = args.output
    else:
        args.process_dir = os.path.join(tempfile.gettempdir(), 'train', args.name)
    log.debug(f'args: {vars(args)}')
    log.debug(f'server flags: {server_options.flags}')
    log.debug(f'server options: {server_options.options}')


async def training_loop():
    async def async_train():
        res = await sdapi.post('/sdapi/v1/train/embedding', options.embedding)
        log.info(f'train embedding result: {res}')

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
    log.info(f'{args.type} options: {options.embedding}')
    create_options = util.Map({
        "name": args.name,
        "num_vectors_per_token": args.dim,
        "overwrite_old": False,
        "init_text": args.tag,
    })
    fn = os.path.join(args.embeddings_dir, args.name) + '.pt'
    if os.path.exists(fn) and args.overwrite:
        log.warning(f'delete existing embedding {fn}')
        os.remove(fn)
    else:
        log.error(f'embedding exists {fn}')
        return
    log.info(f'create embedding {create_options}')
    res = sdapi.postsync('/sdapi/v1/create/embedding', create_options)
    if 'info' in res and 'error' in res['info']: # formatted error
        log.error(res.info)
    elif 'info' in res: # no error
        asyncio.run(training_loop())
    else: # unknown error
        log.error(f'create embedding error {res}')


def train_lora():
    fn = os.path.join(options.lora.output_dir, args.name)
    for ext in ['.ckpt', '.pt', '.safetensors']:
        if os.path.exists(fn + ext):
            if args.overwrite:
                log.warning(f'delete existing lora: {fn + ext}')
                os.remove(fn + ext)
            else:
                log.error(f'lora exists: {fn + ext}')
                return
    log.info(f'{args.type} options: {options.lora}')
    # lora imports
    lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lora'))
    lycoris_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lycoris'))
    sys.path.append(lora_path)
    if args.type == 'lyco':
        sys.path.append(lycoris_path)
    log.debug('importing lora lib')
    import train_network
    train_network.train(options.lora)
    if args.type == 'lyco':
        log.debug('importing lycoris lib')
        import importlib
        _network_module = importlib.import_module(options.lora.network_module)


def prepare_options():
    if args.type == 'embedding':
        log.info('train embedding')
        options.lora.in_json = None
    if args.type == 'dreambooth':
        log.info('train using dreambooth style training')
        options.lora.in_json = None
    if args.type == 'lora':
        log.info('train using lora style training')
        options.lora.output_dir = args.lora_dir
        options.lora.in_json = os.path.join(args.process_dir, args.name + '.json')
    if args.type == 'lyco':
        log.info('train using lycoris network')
        options.lora.output_dir = args.lyco_dir
        options.lora.network_module = 'lycoris.kohya'
        options.lora.in_json = os.path.join(args.process_dir, args.name + '.json')
    # lora specific
    options.lora.pretrained_model_name_or_path = args.model
    options.lora.output_name = args.name
    options.lora.max_train_steps = args.steps
    options.lora.network_dim = args.dim
    options.lora.network_alpha = args.dim // 2 if args.alpha == 0 else args.alpha
    options.lora.netwoork_args = []
    if args.algo is not None:
        options.lora.netwoork_args.append(f'algo={args.algo}')
    if args.args is not None:
        for net_arg in args.args:
            options.lora.netwoork_args.append(net_arg)
    options.lora.gradient_accumulation_steps = args.gradient
    options.lora.learning_rate = args.lr
    options.lora.train_batch_size = args.batch
    options.lora.train_data_dir = args.process_dir
    # embedding specific
    options.embedding.embedding_name = args.name
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
    log.info(f'processing steps: {processing_options}')
    for step in processing_options:
        if step not in valid_steps:
            log.error(f'invalid processing step: {[step]}')
            exit(1)
    for root, _sub_dirs, folder in os.walk(args.input):
        files = [os.path.join(root, f) for f in folder if filetype.is_image(os.path.join(root, f))]
    log.info(f'processing input images: {len(files)}')
    if os.path.exists(args.process_dir):
        if args.overwrite:
            log.warning(f'removing existing processed folder: {args.process_dir}')
            shutil.rmtree(args.process_dir, ignore_errors=True)
        else:
            log.info(f'processed folder exists: {args.process_dir}')
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
        log.info(f'processing current step: {opts}')
        tag = step
        if tag == 'original' and args.tag is not None:
            concept = args.tag.split(',')[0].strip()
        else:
            concept = step
        if args.type in ['lora', 'lyco', 'dreambooth']:
            folder = os.path.join(args.process_dir, str(args.repeats) + '_' + concept) # separate concepts per folder
        if args.type in ['embedding']:
            folder = os.path.join(args.process_dir) # everything into same folder
        log.info(f'processing concept: {concept}')
        log.info(f'processing output folder: {folder}')
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        results = {}
        for f in files:
            res = process.file(filename = f, folder = folder, tag = args.tag, requested = opts)
            if res.image: # valid result
                results[res.type] = results.get(res.type, 0) + 1
                results['total'] = results.get('total', 0) + 1
                rel_path = res.basename.replace(os.path.commonpath([res.basename, args.process_dir]), '')
                if rel_path.startswith(os.path.sep):
                    rel_path = rel_path[1:]
                metadata[rel_path] = { 'caption': res.caption, 'tags': ','.join(res.tags) }
                if options.lora.in_json is None:
                    with open(res.output.replace(options.process.format, '.txt'), "w", encoding='utf-8') as outfile:
                        outfile.write(res.caption)
            log.info(f"processing {'saved' if res.image is not None else 'skipped'}: {f} => {res.output} {res.ops} {res.message}")
    folders = [os.path.join(args.process_dir, folder) for folder in os.listdir(args.process_dir) if os.path.isdir(os.path.join(args.process_dir, folder))]
    log.info(f'input datasets {folders}')
    if options.lora.in_json is not None:
        with open(options.lora.in_json, "w", encoding='utf-8') as outfile: # write json at the end only
            outfile.write(json.dumps(metadata, indent=2))
        for folder in folders: # create latents
            latents.create_vae_latents(util.Map({ 'input': folder, 'json': options.lora.in_json }))
            latents.unload_vae()
    r = { 'inputs': len(files), 'outputs': results, 'metadata': options.lora.in_json }
    log.info(f'processing steps result: {r}')
    if args.gradient < 0:
        log.info(f"setting gradient accumulation to number of images: {results['total']}")
        options.lora.gradient_accumulation_steps = results['total']
        options.embedding.gradient_step = results['total']
    process.unload()


if __name__ == '__main__':
    log.info('SD.Next train script')
    parse_args()
    setup_logging()
    prepare_server()
    verify_args()
    prepare_options()
    mem_stats()
    process_inputs()
    mem_stats()
    try:
        if args.type == 'embedding':
            train_embedding()
        if args.type == 'lora' or args.type == 'lyco' or args.type == 'dreambooth':
            train_lora()
    except KeyboardInterrupt as e:
        log.error('interrupt requested')
        sdapi.interrupt()
    mem_stats()
    log.info('done')
