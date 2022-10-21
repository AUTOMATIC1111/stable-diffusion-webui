import math
import os

import numpy as np
from PIL import Image

import torch
import tqdm

from modules import processing, shared, images, devices, sd_models
from modules.shared import opts
import modules.gfpgan_model
from modules.ui import plaintext_to_html
import modules.codeformer_model
import piexif
import piexif.helper
import gradio as gr


cached_images = {}


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
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
        image_list = [file for file in [os.path.join(input_dir, x) for x in sorted(os.listdir(input_dir))] if os.path.isfile(file)]
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

    
    for image, image_name in zip(imageArr, imageNameArr):
        if image is None:
            return outputs, "Please select an input image.", ''
        existing_pnginfo = image.info or {}

        image = image.convert("RGB")
        info = ""

        if gfpgan_visibility > 0:
            restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
            res = Image.fromarray(restored_img)

            if gfpgan_visibility < 1.0:
                res = Image.blend(image, res, gfpgan_visibility)

            info += f"GFPGAN visibility:{round(gfpgan_visibility, 2)}\n"
            image = res

        if codeformer_visibility > 0:
            restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
            res = Image.fromarray(restored_img)

            if codeformer_visibility < 1.0:
                res = Image.blend(image, res, codeformer_visibility)

            info += f"CodeFormer w: {round(codeformer_weight, 2)}, CodeFormer visibility:{round(codeformer_visibility, 2)}\n"
            image = res

        if resize_mode == 1:
            upscaling_resize = max(upscaling_resize_w/image.width, upscaling_resize_h/image.height)
            crop_info = " (crop)" if upscaling_crop else ""
            info += f"Resize to: {upscaling_resize_w:g}x{upscaling_resize_h:g}{crop_info}\n"

        if upscaling_resize != 1.0:
            def upscale(image, scaler_index, resize, mode, resize_w, resize_h, crop):
                small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
                pixels = tuple(np.array(small).flatten().tolist())
                key = (resize, scaler_index, image.width, image.height, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
                       resize_mode, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop) + pixels

                c = cached_images.get(key)
                if c is None:
                    upscaler = shared.sd_upscalers[scaler_index]
                    c = upscaler.scaler.upscale(image, resize, upscaler.data_path)
                    if mode == 1 and crop:
                        cropped = Image.new("RGB", (resize_w, resize_h))
                        cropped.paste(c, box=(resize_w // 2 - c.width // 2, resize_h // 2 - c.height // 2))
                        c = cropped
                    cached_images[key] = c

                return c

            info += f"Upscale: {round(upscaling_resize, 3)}, model:{shared.sd_upscalers[extras_upscaler_1].name}\n"
            res = upscale(image, extras_upscaler_1, upscaling_resize, resize_mode, upscaling_resize_w, upscaling_resize_h, upscaling_crop)

            if extras_upscaler_2 != 0 and extras_upscaler_2_visibility > 0:
                res2 = upscale(image, extras_upscaler_2, upscaling_resize, resize_mode, upscaling_resize_w, upscaling_resize_h, upscaling_crop)
                info += f"Upscale: {round(upscaling_resize, 3)}, visibility: {round(extras_upscaler_2_visibility, 3)}, model:{shared.sd_upscalers[extras_upscaler_2].name}\n"
                res = Image.blend(res, res2, extras_upscaler_2_visibility)

            image = res

        while len(cached_images) > 2:
            del cached_images[next(iter(cached_images.keys()))]
        
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


def run_pnginfo(image):
    if image is None:
        return '', '', ''

    items = image.info
    geninfo = ''

    if "exif" in image.info:
        exif = piexif.load(image.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        items['exif comment'] = exif_comment
        geninfo = exif_comment

        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
                      'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)

    geninfo = items.get('parameters', geninfo)

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


def run_modelmerger(primary_model_name, secondary_model_name, teritary_model_name, interp_method, multiplier, save_as_half, custom_name):
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    primary_model_info = sd_models.checkpoints_list[primary_model_name]
    secondary_model_info = sd_models.checkpoints_list[secondary_model_name]
    teritary_model_info = sd_models.checkpoints_list.get(teritary_model_name, None)

    print(f"Loading {primary_model_info.filename}...")
    primary_model = torch.load(primary_model_info.filename, map_location='cpu')
    theta_0 = sd_models.get_state_dict_from_checkpoint(primary_model)

    print(f"Loading {secondary_model_info.filename}...")
    secondary_model = torch.load(secondary_model_info.filename, map_location='cpu')
    theta_1 = sd_models.get_state_dict_from_checkpoint(secondary_model)

    if teritary_model_info is not None:
        print(f"Loading {teritary_model_info.filename}...")
        teritary_model = torch.load(teritary_model_info.filename, map_location='cpu')
        theta_2 = sd_models.get_state_dict_from_checkpoint(teritary_model)
    else:
        teritary_model = None
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
    del theta_2, teritary_model

    for key in tqdm.tqdm(theta_0.keys()):
        if 'model' in key and key in theta_1:

            theta_0[key] = theta_func2(theta_0[key], theta_1[key], multiplier)

            if save_as_half:
                theta_0[key] = theta_0[key].half()

    # I believe this part should be discarded, but I'll leave it for now until I am sure
    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:
            theta_0[key] = theta_1[key]
            if save_as_half:
                theta_0[key] = theta_0[key].half()

    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    filename = primary_model_info.model_name + '_' + str(round(1-multiplier, 2)) + '-' + secondary_model_info.model_name + '_' + str(round(multiplier, 2)) + '-' + interp_method.replace(" ", "_") + '-merged.ckpt'
    filename = filename if custom_name == '' else (custom_name + '.ckpt')
    output_modelname = os.path.join(ckpt_dir, filename)

    print(f"Saving to {output_modelname}...")
    torch.save(primary_model, output_modelname)

    sd_models.list_models()

    print(f"Checkpoint saved.")
    return ["Checkpoint saved to " + output_modelname] + [gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)]
