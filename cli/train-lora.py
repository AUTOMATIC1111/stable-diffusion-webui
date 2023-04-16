#!/bin/env python

"""
Extract approximating LoRA by SVD from two SD models
Based on: <https://github.com/kohya-ss/sd-scripts/blob/main/networks/train_network.py>

Train LoRA with custom preprocessing, tagging and bucketing

Disabled/broken:
- `accelerate` with *dynamo* enabled
- `xformers` due to *faketensors* requirement
- `mem_eff_attn` due to *forwardfunc* mismatch
- 'use_8bit_adam` due to *bitsandbyttes* CUDA errors
"""

import os
import re
import gc
import sys
import json
import time
import shutil
import argparse
import tempfile
import torch
import logging
import importlib
import transformers
from pathlib import Path
from modules.util import log, Map, get_memory
import modules.process
import modules.sdapi

latents = importlib.import_module('modules.lora-latents')

lora_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lora'))
sys.path.append(lora_path)
lycoris_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'modules', 'lycoris'))
sys.path.append(lycoris_path)
from train_network import train

options = Map({
    "bucket_no_upscale": False,
    "bucket_reso_steps": 64,
    "cache_latents": True,
    "caption_dropout_every_n_epochs": None,
    "caption_dropout_rate": 0.0,
    "caption_extension": ".txt",
    "caption_extention": ".txt",
    "caption_tag_dropout_rate": 0.0,
    "clip_skip": None,
    "color_aug": False,
    "dataset_repeats": 1,
    "debug_dataset": False,
    "enable_bucket": False,
    "face_crop_aug_range": None,
    "flip_aug": False,
    "full_fp16": False,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False,
    "in_json": "",
    "keep_tokens": None,
    "learning_rate": 5e-05,
    "log_prefix": None,
    "logging_dir": None,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 0,
    "max_bucket_reso": 1024,
    "max_data_loader_n_workers": 8,
    "max_grad_norm": 0.0,
    "max_token_length": None,
    "max_train_epochs": None,
    "max_train_steps": 5000,
    "mem_eff_attn": False,
    "min_bucket_reso": 256,
    "mixed_precision": "fp16",
    "network_alpha": 1.0,
    "network_args": None,
    "network_dim": 16,
    "network_module": "networks.lora",
    "network_train_text_encoder_only": False,
    "network_train_unet_only": False,
    "network_weights": None,
    "no_metadata": False,
    "output_dir": "",
    "output_name": "",
    "persistent_data_loader_workers": False,
    "pretrained_model_name_or_path": "",
    "prior_loss_weight": 1.0,
    "random_crop": False,
    "reg_data_dir": None,
    "resolution": "512,512",
    "resume": None,
    "save_every_n_epochs": None,
    "save_last_n_epochs_state": None,
    "save_last_n_epochs": None,
    "save_model_as": "ckpt",
    "save_n_epoch_ratio": None,
    "save_precision": "fp16",
    "save_state": False,
    "seed": 42,
    "shuffle_caption": False,
    "text_encoder_lr": 5e-05,
    "train_batch_size": 1,
    "train_data_dir": "",
    "training_comment": "mood-magic",
    "unet_lr": 0.001,
    "use_8bit_adam": False,
    "v_parameterization": False,
    "v2": False,
    "vae": None,
    "xformers": False,
})


def mem_stats():
    gc.collect()
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    mem = get_memory()
    log.info({ 'memory': { 'ram': mem.ram, 'gpu': mem.gpu } })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train lora')
    parser.add_argument('--model', type=str, default=None, required=False, help='original model to use a base for training, default: active model')
    parser.add_argument('--input', '--dataset', type=str, default=None, required=True, help='input folder with training images')
    parser.add_argument('--output', '--lora', type=str, default=None, required=True, help='lora name')
    parser.add_argument('--tag', type=str, default=None, required=False, help='primary tag')
    parser.add_argument('--dir', type=str, default=None, required=False, help='folder containing lora checkpoints')
    parser.add_argument('--interim', type=int, default=0, help = 'save interim checkpoints after n epoch')
    parser.add_argument('--process', type=str, default='original', required=False, help='list of processing steps: original,face,body,blur,range,upscale,restore')
    parser.add_argument('--noprocess', default = False, action='store_true', help = 'skip processing and use existing input data')
    parser.add_argument('--notrain', default = False, action='store_true', help = 'just run processing and skip training')
    parser.add_argument('--nocaptions', default = False, action='store_true', help = 'skip creating captions and tags')
    parser.add_argument('--nolatents', default = False, action='store_true', help = 'skip generating vae latents')
    parser.add_argument('--offline', default = False, action='store_true', help = 'do not use webui server for processing')
    parser.add_argument('--shutdown', default = False, action='store_true', help = 'shutdown webui server')
    parser.add_argument('--gradient', type=int, default=1, required=False, help='gradient accumulation steps, default: %(default)s')
    parser.add_argument('--steps', type=int, default=4000, required=False, help='training steps, default: %(default)s')
    parser.add_argument('--dim', type=int, default=40, required=False, help='network dimension, default: %(default)s')
    parser.add_argument('--repeats', type=int, default=10, required=False, help='number of repeats per image, default: %(default)s')
    parser.add_argument('--alpha', type=float, default=0, required=False, help='alpha for weights scaling, default: half of dim')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size, default: %(default)s')
    parser.add_argument('--lr', type=float, default=1e-04, required=False, help='model learning rate, default: %(default)s')
    parser.add_argument('--unetlr', type=float, default=1e-04, required=False, help='unet learning rate, default: %(default)s')
    parser.add_argument('--textlr', type=float, default=5e-05, required=False, help='text encoder learning rate, default: %(default)s')
    parser.add_argument('--dreambooth', default=False, action='store_true', help = "use dreambooth style training")
    parser.add_argument('--lycoris', default=False, action='store_true', help = "use lycoris style training")
    parser.add_argument('--debug', default=False, action='store_true', help = "enable debug logging")
    args = parser.parse_args()
    defaults = Map({ 'options': {}, 'flags': {} }) if args.offline else Map(modules.sdapi.options())

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    if args.model is None:
        args.model = defaults.options.get('sd_model_checkpoint', None)
        args.model = args.model.split(' [')[0] if args.model is not None else None
    if args.dir is None:
        args.dir = defaults.flags.get('lora_dir', None)
    if not os.path.isabs(args.model) and args.dir is not None and not os.path.exists(args.model):
        args.model = os.path.abspath(os.path.join(args.dir, os.pardir, 'Stable-diffusion', args.model))
    if args.dir is None:
        args.dir = os.path.join(args.input, 'lora')
    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        log.error({ 'lora cannot find model': args.model })
        exit(1)
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        log.error({ 'lora cannot find training dir': args.input })
        exit(1)
    if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
        log.error({ 'lora cannot find training dir': args.dir })
        exit(1)
    options.pretrained_model_name_or_path = args.model
    options.output_dir = args.dir
    options.output_name = args.output
    options.max_train_steps = args.steps
    options.network_dim = args.dim
    options.network_alpha = args.dim // 2 if args.alpha == 0 else args.alpha
    options.gradient_accumulation_steps = args.gradient
    options.save_every_n_epochs = args.interim if args.interim > 0 else None
    options.learning_rate = args.lr
    options.unet_lr = args.unetlr
    options.text_encoder_lr = args.textlr
    options.train_batch_size = args.batch
    log.info({ 'train lora args': vars(options) })
    transformers.logging.set_verbosity_error()
    mem_stats()

    json_file = os.path.join(tempfile.gettempdir(), args.output, args.output + '.json')
    base = os.path.join(tempfile.gettempdir(), args.output)
    options.train_data_dir = base
    res = None

    if args.dreambooth:
        log.info({ 'using dreambooth style training': True })
        options.in_json = None
    else:
        options.in_json = json_file

    for root, _sub_dirs, folder in os.walk(args.input):
        files = [os.path.join(root, f) for f in folder]

    if not args.noprocess:
        # preprocess
        processing_options = args.process.split(',')
        processing_options = [opt.strip() for opt in re.split(',| ', args.process)]
        log.info({ 'processing steps': processing_options })

        if os.path.exists(json_file):
            os.remove(json_file)

        steps = [step for step in processing_options if step in ['face', 'body', 'original']]
        for step in steps:
            # processing_options = [step for step in processing_options if step not in ['face', 'body', 'original']].append(step)
            if step == 'face':
                opts = [step for step in processing_options if step not in ['body', 'original']]
            if step == 'body':
                opts = [step for step in processing_options if step not in ['face', 'original', 'upscale', 'restore']]
            if step == 'original':
                opts = [step for step in processing_options if step not in ['face', 'body', 'upscale', 'restore', 'blur', 'range']]
            log.info({ 'processing step': opts })
            concept = step
            if concept == 'original' and args.tag is not None:
                concept = args.tag.split(',')[0].strip()
            dir = os.path.join(base, str(args.repeats) + '_' + concept)
            if os.path.exists(dir):
                shutil.rmtree(dir, ignore_errors=True)
            Path(dir).mkdir(parents=True, exist_ok=True)

            for f in files:
                try:
                    res, metadata = modules.process.process_file(f = f, dst = dir, preview = False, offline = args.offline, txt = args.dreambooth, tag = args.tag, opts = opts)
                    if not args.dreambooth:
                        with open(json_file, "w") as outfile:
                            outfile.write(json.dumps(metadata, indent=2))
                except ValueError as e:
                    exit(1)
            log.info({ 'processed step': step, 'outputs': res, 'inputs': len(files), 'metadata': json_file, 'path': dir })

        modules.process.unload_models()
        mem_stats()


    dirs = [os.path.join(base, dir) for dir in os.listdir(base) if os.path.isdir(os.path.join(base, dir))]
    log.info({ 'input datasets': dirs, 'metadata': json_file })

    if not args.nolatents and not args.dreambooth:
        # create latents
        for dir in dirs:
            latents.create_vae_latents(Map({ 'input': dir, 'json': json_file }))
            latents.unload_vae()
        mem_stats()
    else:
        log.info({ 'skip processing': len(files), 'metadata': json_file, 'path': dir })

    if args.shutdown:
        log.info({ 'server shutdown required': True })
        modules.sdapi.shutdown()
        time.sleep(1)

    if args.lycoris:
        log.info({ 'using lycoris network': True })
        options.network_module = 'lycoris.kohya'
    if not args.notrain:
        train(options)
        mem_stats()
