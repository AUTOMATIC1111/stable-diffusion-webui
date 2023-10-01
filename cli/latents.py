#!/usr/bin/env python

import os
import sys
import json
import pathlib
import argparse
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from util import Map

from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console

console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
pretty_install(console=console)
traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules', 'lora'))
import library.model_util as model_util
import library.train_util as train_util

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
options = Map({
  'batch': 1,
  'input': '',
  'json': '',
  'max': 1024,
  'min': 256,
  'noupscale': False,
  'precision': 'fp32',
  'resolution': '512,512',
  'steps': 64,
  'vae': 'stabilityai/sd-vae-ft-mse'
})
vae = None


def get_latents(local_vae, images, weight_dtype):
    image_transforms = transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ])
    img_tensors = [image_transforms(image) for image in images]
    img_tensors = torch.stack(img_tensors)
    img_tensors = img_tensors.to(device, weight_dtype)
    with torch.no_grad():
        latents = local_vae.encode(img_tensors).latent_dist.sample().float().to('cpu').numpy()
    return latents, [images[0].shape[0], images[0].shape[1]]


def get_npz_filename_wo_ext(data_dir, image_key):
    return os.path.join(data_dir, os.path.splitext(os.path.basename(image_key))[0])


def create_vae_latents(local_params):
    args = Map({**options, **local_params})
    console.log(f'create vae latents args: {args}')
    image_paths = train_util.glob_images(args.input)
    if os.path.exists(args.json):
        with open(args.json, 'rt', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        return
    if args.precision == 'fp16':
        weight_dtype = torch.float16
    elif args.precision == 'bf16':
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    global vae # pylint: disable=global-statement
    if vae is None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        vae.eval()
        vae.to(device, dtype=weight_dtype)
    max_reso = tuple([int(t) for t in args.resolution.split(',')])
    assert len(max_reso) == 2, f'illegal resolution: {args.resolution}'
    bucket_manager = train_util.BucketManager(args.noupscale, max_reso, args.min, args.max, args.steps)
    if not args.noupscale:
        bucket_manager.make_buckets()
    img_ar_errors = []
    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch:
                latents, original_size = get_latents(vae, [img for _, img in bucket], weight_dtype)
                assert latents.shape[2] == bucket[0][1].shape[0] // 8 and latents.shape[3] == bucket[0][1].shape[1] // 8, f'latent shape {latents.shape}, {bucket[0][1].shape}'
                for (image_key, _), latent in zip(bucket, latents):
                    npz_file_name = get_npz_filename_wo_ext(args.input, image_key)
                    # np.savez(npz_file_name, latent)
                    kwargs = {}
                    np.savez(
                        npz_file_name,
                        latents=latent,
                        original_size=np.array(original_size),
                        crop_ltrb=np.array([0, 0]),
                        **kwargs,
                    )
                bucket.clear()
    data = [[(None, ip)] for ip in image_paths]
    bucket_counts = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry[0] is None:
            continue
        img_tensor, image_path = data_entry[0]
        if img_tensor is not None:
            image = transforms.functional.to_pil_image(img_tensor)
        else:
            image = Image.open(image_path)
        image_key = os.path.basename(image_path)
        image_key = os.path.join(os.path.basename(pathlib.Path(image_path).parent), pathlib.Path(image_path).stem)
        if image_key not in metadata:
            metadata[image_key] = {}
        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1
        metadata[image_key]['train_resolution'] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)
        if not args.noupscale:
            assert resized_size[0] == reso[0] or resized_size[1] == reso[1], f'internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}'
            assert resized_size[0] >= reso[0] and resized_size[1] >= reso[1], f'internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}'
        assert resized_size[0] >= reso[0] and resized_size[1] >= reso[1], f'internal error resized size is small: {resized_size}, {reso}'
        image = np.array(image)
        if resized_size[0] != image.shape[1] or resized_size[1] != image.shape[0]:
            image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
        if resized_size[0] > reso[0]:
            trim_size = resized_size[0] - reso[0]
            image = image[:, trim_size//2:trim_size//2 + reso[0]]
        if resized_size[1] > reso[1]:
            trim_size = resized_size[1] - reso[1]
            image = image[trim_size//2:trim_size//2 + reso[1]]
        assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f'internal error, illegal trimmed size: {image.shape}, {reso}'
        bucket_manager.add_image(reso, (image_key, image))
        process_batch(False)

    process_batch(True)
    vae.to('cpu')

    bucket_manager.sort()
    img_ar_errors = np.array(img_ar_errors)
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            console.log(f'vae latents bucket: {i+1}/{len(bucket_manager.resos)} resolution: {reso} images: {count} mean-ar-error: {np.mean(img_ar_errors)}')
    with open(args.json, 'wt', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def unload_vae():
    global vae # pylint: disable=global-statement
    vae = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='directory for train images')
    parser.add_argument('--json', type=str, required=True, help='metadata file to input')
    parser.add_argument('--vae', type=str, required=True, help='model name or path to encode latents')
    parser.add_argument('--batch', type=int, default=1, help='batch size in inference')
    parser.add_argument('--resolution', type=str, default='512,512', help='max resolution in fine tuning (width,height)')
    parser.add_argument('--min', type=int, default=256, help='minimum resolution for buckets')
    parser.add_argument('--max', type=int, default=1024, help='maximum resolution for buckets')
    parser.add_argument('--steps', type=int, default=64, help='steps of resolution for buckets, divisible by 8')
    parser.add_argument('--noupscale', action='store_true', help='make bucket for each image without upscaling')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], help='use precision')
    params = parser.parse_args()
    create_vae_latents(vars(params))
