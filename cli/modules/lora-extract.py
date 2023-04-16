#!/bin/env python

"""
Extract approximating LoRA by SVD from two SD models
Based on: <https://github.com/kohya-ss/sd-scripts/blob/main/networks/extract_lora_from_models.py>
"""

import os
import sys
import time
import argparse
import torch
import transformers
from tqdm import tqdm
from util import log

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'modules', 'lora'))
import library.model_util as model_util
import networks.lora as lora


def svd(args): # pylint: disable=redefined-outer-name
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    transformers.logging.set_verbosity_error()
    CLAMP_QUANTILE = 0.99
    MIN_DIFF = 1e-6
    if args.precision == 'fp32':
        save_dtype = torch.float
    elif args.precision == 'fp16':
        save_dtype = torch.float16
    elif args.precision == 'bf16':
        save_dtype = torch.bfloat16
    else:
        save_dtype = None
    t0 = time.time()
    log.info({ 'loading model': args.original })
    text_encoder_o, _, unet_o = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.original)
    log.info({ 'loading model': args.tuned })
    text_encoder_t, _, unet_t = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.tuned)
    with torch.no_grad():
        torch.cuda.empty_cache()
    # create LoRA network to extract weights: Use dim (rank) as alpha
    lora_network_o = lora.create_network(1.0, args.dim, args.dim, None, text_encoder_o, unet_o)
    lora_network_t = lora.create_network(1.0, args.dim, args.dim, None, text_encoder_t, unet_t)
    assert len(lora_network_o.text_encoder_loras) == len(lora_network_t.text_encoder_loras), 'model version is different'
    # get diffs
    diffs = {}
    text_encoder_different = False
    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight - module_o.weight
        # Text Encoder might be same
        if torch.max(torch.abs(diff)) > MIN_DIFF:
            text_encoder_different = True
        diff = diff.float()
        diffs[lora_name] = diff

    if not text_encoder_different:
        log.info({ 'lora': 'text encoder is same, extract U-Net only' })
        lora_network_o.text_encoder_loras = []
        diffs = {}

    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight - module_o.weight
        diff = diff.float()
        diff = diff.to(device)
        diffs[lora_name] = diff
    t1 = time.time()
    log.info({ 'lora models': 'ready', 'time': round(t1 - t0, 2) })

    # make LoRA with svd
    log.info({ 'lora': 'calculating by svd' })
    rank = args.dim
    lora_weights = {}
    with torch.no_grad():
        for lora_name, mat in tqdm(list(diffs.items())):
            conv2d = len(mat.size()) == 4
            if conv2d:
                mat = mat.squeeze()
            U, S, Vh = torch.linalg.svd(mat)
            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)
            Vh = Vh[:rank, :]
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
            lora_weights[lora_name] = (U, Vh)
    t2 = time.time()

    # make state dict for LoRA
    lora_network_o.apply_to(text_encoder_o, unet_o, text_encoder_different, True)
    lora_sd = lora_network_o.state_dict()
    log.info({ 'lora extracted weights': len(lora_sd), 'time': round(t2 - t1, 2) })

    for key in list(lora_sd.keys()):
        if 'alpha' in key:
            continue
        lora_name = key.split('.')[0]
        i = 0 if 'lora_up' in key else 1
        weights = lora_weights[lora_name][i]
        # print(key, i, weights.size(), lora_sd[key].size())
        if len(lora_sd[key].size()) == 4: # pylint: disable=unsubscriptable-object
            weights = weights.unsqueeze(2).unsqueeze(3)
        assert weights.size() == lora_sd[key].size(), f'size unmatch: {key}' # pylint: disable=unsubscriptable-object
        lora_sd[key] = weights  # pylint: disable=unsupported-assignment-operation

    # load state dict to LoRA and save it
    info = lora_network_o.load_state_dict(lora_sd)
    log.info({ 'lora loading extracted weights': info })

    dir_name = os.path.dirname(args.save)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # minimum metadata
    metadata = {'ss_network_dim': str(args.dim), 'ss_network_alpha': str(args.dim)}
    lora_network_o.save_weights(args.save, save_dtype, metadata)
    t3 = time.time()
    log.info({ 'lora saved weights': args.save, 'time': round(t3 - t2, 2) })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'extract lora weights')
    parser.add_argument('--v2', action='store_true', help='load Stable Diffusion v2.x model / Stable Diffusion')
    parser.add_argument('--precision', type=str, default='fp16', choices=[None, 'fp32', 'fp16', 'bf16'], help='precision in saving, same to merging if omitted')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu or cuda if available')
    parser.add_argument('--original', type=str, default=None, required=True, help='Stable Diffusion original model: ckpt or safetensors file')
    parser.add_argument('--tuned', type=str, default=None, required=True, help='Stable Diffusion tuned model, LoRA is difference of `original to tuned`: ckpt or safetensors file')
    parser.add_argument('--save', type=str, default=None, required=True, help='destination file name: ckpt or safetensors file')
    parser.add_argument('--dim', type=int, default=4, help='dimension (rank) of LoRA')
    args = parser.parse_args()
    log.info({ 'extract lora args': vars(args) })
    if not os.path.exists(args.original) or not os.path.exists(args.tuned):
        log.error({ 'models not found': [args.original, args.tuned] })
    else:
        svd(args)
