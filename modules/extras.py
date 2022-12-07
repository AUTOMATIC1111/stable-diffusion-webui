from __future__ import annotations
import math
import os
import sys
import traceback

import numpy as np
from PIL import Image

import torch
import tqdm

from typing import Callable, List, OrderedDict, Tuple
from functools import partial
from dataclasses import dataclass

from modules import processing, shared, images, devices, sd_models, sd_samplers
from modules.shared import opts
import modules.gfpgan_model
from modules.ui import plaintext_to_html
import modules.codeformer_model
import piexif
import piexif.helper
import gradio as gr
import safetensors.torch

class LruCache(OrderedDict):
    @dataclass(frozen=True)
    class Key:
        image_hash: int
        info_hash: int
        args_hash: int

    @dataclass
    class Value:
        image: Image.Image
        info: str

    def __init__(self, max_size: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_size = max_size

    def get(self, key: LruCache.Key) -> LruCache.Value:
        ret = super().get(key)
        if ret is not None:
            self.move_to_end(key)  # Move to end of eviction list
        return ret

    def put(self, key: LruCache.Key, value: LruCache.Value) -> None:
        self[key] = value
        while len(self) > self._max_size:
            self.popitem(last=False)


cached_images: LruCache = LruCache(max_size=5)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool):
    devices.torch_gc()

    imageArr = []
    # Also keep track of original file names
    imageNameArr = []
    outputs = []
    
    if extras_mode == 1:
        #convert file to pillow image
        for img in image_folder:
            image = Image.open(img)
            imageArr.append(image)
            imageNameArr.append(os.path.splitext(img.orig_name)[0])
    elif extras_mode == 2:
        assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'

        if input_dir == '':
            return outputs, "Please select an input directory.", ''
        image_list = shared.listfiles(input_dir)
        for img in image_list:
            try:
                image = Image.open(img)
            except Exception:
                continue
            imageArr.append(image)
            imageNameArr.append(img)
    else:
        imageArr.append(image)
        imageNameArr.append(None)

    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    # Extra operation definitions

    def run_gfpgan(image: Image.Image, info: str) -> Tuple[Image.Image, str]:
        restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(image, res, gfpgan_visibility)

        info += f"GFPGAN visibility:{round(gfpgan_visibility, 2)}\n"
        return (res, info)

    def run_codeformer(image: Image.Image, info: str) -> Tuple[Image.Image, str]:
        restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(image, res, codeformer_visibility)

        info += f"CodeFormer w: {round(codeformer_weight, 2)}, CodeFormer visibility:{round(codeformer_visibility, 2)}\n"
        return (res, info)

    def upscale(image, scaler_index, resize, mode, resize_w, resize_h, crop):
        upscaler = shared.sd_upscalers[scaler_index]
        res = upscaler.scaler.upscale(image, resize, upscaler.data_path)
        if mode == 1 and crop:
            cropped = Image.new("RGB", (resize_w, resize_h))
            cropped.paste(res, box=(resize_w // 2 - res.width // 2, resize_h // 2 - res.height // 2))
            res = cropped
        return res

    def run_prepare_crop(image: Image.Image, info: str) -> Tuple[Image.Image, str]:
        # Actual crop happens in run_upscalers_blend, this just sets upscaling_resize and adds info text
        nonlocal upscaling_resize
        if resize_mode == 1:
            upscaling_resize = max(upscaling_resize_w/image.width, upscaling_resize_h/image.height)
            crop_info = " (crop)" if upscaling_crop else ""
            info += f"Resize to: {upscaling_resize_w:g}x{upscaling_resize_h:g}{crop_info}\n"
        return (image, info)

    @dataclass
    class UpscaleParams:
        upscaler_idx: int
        blend_alpha: float

    def run_upscalers_blend(params: List[UpscaleParams], image: Image.Image, info: str) -> Tuple[Image.Image, str]:
        blended_result: Image.Image = None
        image_hash: str = hash(np.array(image.getdata()).tobytes())
        for upscaler in params:
            upscale_args = (upscaler.upscaler_idx, upscaling_resize, resize_mode,
                            upscaling_resize_w, upscaling_resize_h, upscaling_crop)
            cache_key = LruCache.Key(image_hash=image_hash,
                                     info_hash=hash(info),
                                     args_hash=hash(upscale_args))
            cached_entry = cached_images.get(cache_key)
            if cached_entry is None:
                res = upscale(image, *upscale_args)
                info += f"Upscale: {round(upscaling_resize, 3)}, visibility: {upscaler.blend_alpha}, model:{shared.sd_upscalers[upscaler.upscaler_idx].name}\n"
                cached_images.put(cache_key, LruCache.Value(image=res, info=info))
            else:
                res, info = cached_entry.image, cached_entry.info

            if blended_result is None:
                blended_result = res
            else:
                blended_result = Image.blend(blended_result, res, upscaler.blend_alpha)
        return (blended_result, info)

    # Build a list of operations to run
    facefix_ops: List[Callable] = []
    facefix_ops += [run_gfpgan] if gfpgan_visibility > 0 else []
    facefix_ops += [run_codeformer] if codeformer_visibility > 0 else []

    upscale_ops: List[Callable] = []
    upscale_ops += [run_prepare_crop] if resize_mode == 1 else []

    if upscaling_resize != 0:
        step_params: List[UpscaleParams] = []
        step_params.append(UpscaleParams(upscaler_idx=extras_upscaler_1, blend_alpha=1.0))
        if extras_upscaler_2 != 0 and extras_upscaler_2_visibility > 0:
            step_params.append(UpscaleParams(upscaler_idx=extras_upscaler_2, blend_alpha=extras_upscaler_2_visibility))

        upscale_ops.append(partial(run_upscalers_blend, step_params))

    extras_ops: List[Callable] = (upscale_ops + facefix_ops) if upscale_first else (facefix_ops + upscale_ops)

    for image, image_name in zip(imageArr, imageNameArr):
        if image is None:
            return outputs, "Please select an input image.", ''
        existing_pnginfo = image.info or {}

        image = image.convert("RGB")
        info = ""
        # Run each operation on each image
        for op in extras_ops:
            image, info = op(image, info)

        if opts.use_original_name_batch and image_name != None:
            basename = os.path.splitext(os.path.basename(image_name))[0]
        else:
            basename = ''

        images.save_image(image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,
                          no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=None)

        if opts.enable_pnginfo:
            image.info = existing_pnginfo
            image.info["extras"] = info

        if extras_mode != 2 or show_extras_results :
            outputs.append(image)

    devices.torch_gc()

    return outputs, plaintext_to_html(info), ''

def clear_cache():
    cached_images.clear()


def run_pnginfo(image):
    if image is None:
        return '', '', ''

    geninfo, items = images.read_info_from_image(image)
    items = {**{'parameters': geninfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', geninfo, info


def run_modelmerger(primary_model_name, secondary_model_name, teritary_model_name, interp_method, multiplier, save_as_half, custom_name, checkpoint_format):
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    primary_model_info = sd_models.checkpoints_list[primary_model_name]
    secondary_model_info = sd_models.checkpoints_list[secondary_model_name]
    teritary_model_info = sd_models.checkpoints_list.get(teritary_model_name, None)
    result_is_inpainting_model = False

    print(f"Loading {primary_model_info.filename}...")
    theta_0 = sd_models.read_state_dict(primary_model_info.filename, map_location='cpu')

    print(f"Loading {secondary_model_info.filename}...")
    theta_1 = sd_models.read_state_dict(secondary_model_info.filename, map_location='cpu')

    if teritary_model_info is not None:
        print(f"Loading {teritary_model_info.filename}...")
        theta_2 = sd_models.read_state_dict(teritary_model_info.filename, map_location='cpu')
    else:
        theta_2 = None

    theta_funcs = {
        "Weighted sum": (None, weighted_sum),
        "Add difference": (get_difference, add_difference),
    }
    theta_func1, theta_func2 = theta_funcs[interp_method]

    print(f"Merging...")

    if theta_func1:
        for key in tqdm.tqdm(theta_1.keys()):
            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])
    del theta_2

    for key in tqdm.tqdm(theta_0.keys()):
        if 'model' in key and key in theta_1:
            a = theta_0[key]
            b = theta_1[key]

            # this enables merging an inpainting model (A) with another one (B);
            # where normal model would have 4 channels, for latenst space, inpainting model would
            # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                if a.shape[1] == 4 and b.shape[1] == 9:
                    raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")

                assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"

                theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                result_is_inpainting_model = True
            else:
                theta_0[key] = theta_func2(a, b, multiplier)

            if save_as_half:
                theta_0[key] = theta_0[key].half()

    # I believe this part should be discarded, but I'll leave it for now until I am sure
    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:
            theta_0[key] = theta_1[key]
            if save_as_half:
                theta_0[key] = theta_0[key].half()

    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    filename = \
        primary_model_info.model_name + '_' + str(round(1-multiplier, 2)) + '-' + \
        secondary_model_info.model_name + '_' + str(round(multiplier, 2)) + '-' + \
        interp_method.replace(" ", "_") + \
        '-merged.' +  \
        ("inpainting." if result_is_inpainting_model else "") + \
        checkpoint_format

    filename = filename if custom_name == '' else (custom_name + '.' + checkpoint_format)

    output_modelname = os.path.join(ckpt_dir, filename)

    print(f"Saving to {output_modelname}...")

    _, extension = os.path.splitext(output_modelname)
    if extension.lower() == ".safetensors":
        safetensors.torch.save_file(theta_0, output_modelname, metadata={"format": "pt"})
    else:
        torch.save(theta_0, output_modelname)

    sd_models.list_models()

    print(f"Checkpoint saved.")
    return ["Checkpoint saved to " + output_modelname] + [gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)]
