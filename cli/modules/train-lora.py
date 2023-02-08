#!/bin/env python

"""
Extract approximating LoRA by SVD from two SD models
Based on: <https://github.com/kohya-ss/sd-scripts/blob/main/networks/train_network.py>
"""

import os
import sys
import argparse
import tempfile
import transformers
from pathlib import Path
from util import log, Map
from process import process_file

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'lora'))
from train_network import train


options = Map({
    "v2": False,
    "v_parameterization": False,
    "pretrained_model_name_or_path": "/mnt/d/Models/stable-diffusion/sd-v15-runwayml.ckpt",
    "train_data_dir": "/tmp/rreid/img",
    "shuffle_caption": False,
    "caption_extension": ".txt",
    "caption_extention": None,
    "keep_tokens": None,
    "color_aug": False,
    "flip_aug": False,
    "face_crop_aug_range": None,
    "random_crop": False,
    "debug_dataset": False,
    "resolution": "512,512",
    "cache_latents": True,
    "enable_bucket": False,
    "min_bucket_reso": 256,
    "max_bucket_reso": 1024,
    "bucket_reso_steps": 64,
    "bucket_no_upscale": False,
    "reg_data_dir": None,
    "in_json": "/tmp/rreid/rreid.json",
    "dataset_repeats": 1,
    "output_dir": "/mnt/d/Models/lora/",
    "output_name": "lora-rreid-random-v1",
    "save_precision": "fp16",
    "save_every_n_epochs": 1,
    "save_n_epoch_ratio": None,
    "save_last_n_epochs": None,
    "save_last_n_epochs_state": None,
    "save_state": False,
    "resume": None,
    "train_batch_size": 1,
    "max_token_length": None,
    "use_8bit_adam": False,
    "mem_eff_attn": False,
    "xformers": False,
    "vae": None,
    "learning_rate": 1e-05,
    "max_train_steps": 5000,
    "max_train_epochs": None,
    "max_data_loader_n_workers": 8,
    "persistent_data_loader_workers": False,
    "seed": 42,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "fp16",
    "full_fp16": False,
    "clip_skip": None,
    "logging_dir": None,
    "log_prefix": None,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 0,
    "prior_loss_weight": 1.0,
    "no_metadata": False,
    "save_model_as": "ckpt",
    "unet_lr": 0.001,
    "text_encoder_lr": 5e-05,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1,
    "network_weights": None,
    "network_module": "networks.lora",
    "network_dim": 16,
    "network_alpha": 1.0,
    "network_args": None,
    "network_train_unet_only": False,
    "network_train_text_encoder_only": False,
    "training_comment": "mood-magic"
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train lora')
    parser.add_argument('--model', type=str, default=None, required=True, help='original model to use a base for training')
    parser.add_argument('--input', type=str, default=None, required=True, help='input folder with training images')
    parser.add_argument('--dir', type=str, default=None, required=True, help='folder containing lora checkpoints')
    parser.add_argument('--name', type=str, default=None, required=True, help='lora name')
    parser.add_argument('--steps', type=int, default=5000, required=False, help='training steps')
    parser.add_argument('--dim', type=int, default=16, required=False, help='network dimension')
    parser.add_argument("--noprocess", default = False, action='store_true', help = "skip processing and use existing input data")
    args = parser.parse_args()
    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        log.error({ 'lora cannot find model': args.model })
        exit(1)
    options.pretrained_model_name_or_path = args.model
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        log.error({ 'lora cannot find training dir': args.input })
        exit(1)
    if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
        log.error({ 'lora cannot find training dir': args.dir })
        exit(1)
    options.output_dir = args.dir
    options.output_name = args.name
    options.max_train_steps = args.steps
    options.network_dim = args.dim
    log.info({ 'train lora args': vars(options) })
    transformers.logging.set_verbosity_error()

    if args.noprocess:
        options.train_data_dir = args.input
    else:
        dir = os.path.join(tempfile.gettempdir(), args.name, '10_processed')
        Path(dir).mkdir(parents=True, exist_ok=True)
        files = []
        json_data = {}
        for root, _sub_dirs, folder in os.walk(args.input):
            for f in folder:
                files.append(os.path.join(root, f))
        for f in files:
            res = process_file(f = f, dst = dir, preview = False, offline = True)
            
        log.info({ 'processed': res, 'inputs': len(files) })
        options.train_data_dir = args.input
        dir = os.path.join(tempfile.gettempdir(), args.name)

    train(options)


"""
- cannot use `accelerate` with *dynamo* enabled
- cannot use `xformers` due to *faketensors* requirement
- cannot use `mem_eff_attn` due to *forwardfunc* mismatch

TODO

--gradient_checkpointing
--gradient_accumulation_steps=10
--caption_extension=txt
--in_json

WORKING

process.py --output "/tmp/rreid/img/10_processed" /home/vlado/generative/Input/ryanreid/random --offline

accelerate launch --no_python --quiet --num_cpu_threads_per_process=16 python /home/vlado/dev/automatic/modules/lora/train_network.py \
--pretrained_model_name_or_path="/mnt/d/Models/stable-diffusion/sd-v15-runwayml.ckpt" \
--train_data_dir="/tmp/rreid/img" \
--logging_dir="/tmp/rreid/logging" \
--output_dir="/mnt/d/Models/lora/" \
--output_name="lora-rreid-random-v1" \
--resolution=512,512 \
--learning_rate=1e-5 \
--unet_lr=1e-3 \
--text_encoder_lr=5e-5 \
--lr_scheduler_num_cycles=1 \
--lr_scheduler=cosine \
--max_train_steps=5000 \
--network_alpha=1 \
--network_dim=16 \
--network_module=networks.lora \
--save_every_n_epochs=1 \
--save_model_as=ckpt \
--save_precision=fp16 \
--mixed_precision=fp16 \
--seed=42 \
--train_batch_size=1 \
--cache_latents \

metadata { image_key: img_md: { caption: str, tags: [] } }

abs_path = glob_images(train_data_dir, image_key)

}}

./train-lora.py --model /mnt/d/Models/stable-diffusion/sd-v15-runwayml.ckpt --name rreid --dir /mnt/d/Models/lora --input ~/generative/Input/ryanreid/random/
"""