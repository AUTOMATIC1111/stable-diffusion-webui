from __future__ import annotations
import os

import numpy as np
from PIL import Image

from typing import Callable, List, OrderedDict, Tuple
from functools import partial
from dataclasses import dataclass

from modules import shared, images, devices, ui_components
from modules.shared import opts
import modules.gfpgan_model
import modules.codeformer_model


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


def run_postprocessing(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin()
    shared.state.job = 'extras'

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
        shared.state.job = 'extras-gfpgan'
        restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(image, res, gfpgan_visibility)

        info += f"GFPGAN visibility:{round(gfpgan_visibility, 2)}\n"
        return (res, info)

    def run_codeformer(image: Image.Image, info: str) -> Tuple[Image.Image, str]:
        shared.state.job = 'extras-codeformer'
        restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(image, res, codeformer_visibility)

        info += f"CodeFormer w: {round(codeformer_weight, 2)}, CodeFormer visibility:{round(codeformer_visibility, 2)}\n"
        return (res, info)

    def upscale(image, scaler_index, resize, mode, resize_w, resize_h, crop):
        shared.state.job = 'extras-upscale'
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

        shared.state.textinfo = f'Processing image {image_name}'
        
        existing_pnginfo = image.info or {}

        image = image.convert("RGB")
        info = ""
        # Run each operation on each image
        for op in extras_ops:
            image, info = op(image, info)

        if opts.use_original_name_batch and image_name is not None:
            basename = os.path.splitext(os.path.basename(image_name))[0]
        else:
            basename = ''

        if opts.enable_pnginfo: # append info before save
            image.info = existing_pnginfo
            image.info["extras"] = info

        if save_output:
            # Add upscaler name as a suffix.
            suffix = f"-{shared.sd_upscalers[extras_upscaler_1].name}" if shared.opts.use_upscaler_name_as_suffix else ""
            # Add second upscaler if applicable.
            if suffix and extras_upscaler_2 and extras_upscaler_2_visibility:
                suffix += f"-{shared.sd_upscalers[extras_upscaler_2].name}"

            images.save_image(image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,
                            no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=None, suffix=suffix)

        if extras_mode != 2 or show_extras_results :
            outputs.append(image)

    devices.torch_gc()

    return outputs, ui_components.plaintext_to_html(info), ''


def clear_cache():
    cached_images.clear()

