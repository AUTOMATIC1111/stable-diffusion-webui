import os

import numpy as np
from PIL import Image

import torch
import tqdm

from modules import processing, shared, images, devices
from modules.shared import opts
import modules.gfpgan_model
from modules.ui import plaintext_to_html
import modules.codeformer_model
import piexif
import piexif.helper


cached_images = {}


def run_extras(extras_mode, image, image_folder, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    devices.torch_gc()

    imageArr = []
    # Also keep track of original file names
    imageNameArr = []

    if extras_mode == 1:
        #convert file to pillow image
        for img in image_folder:
            image = Image.fromarray(np.array(Image.open(img)))
            imageArr.append(image)
            imageNameArr.append(os.path.splitext(img.orig_name)[0])
    else:
        imageArr.append(image)
        imageNameArr.append(None)

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    outputs = []
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

        if upscaling_resize != 1.0:
            def upscale(image, scaler_index, resize):
                small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
                pixels = tuple(np.array(small).flatten().tolist())
                key = (resize, scaler_index, image.width, image.height, gfpgan_visibility, codeformer_visibility, codeformer_weight) + pixels

                c = cached_images.get(key)
                if c is None:
                    upscaler = shared.sd_upscalers[scaler_index]
                    c = upscaler.upscale(image, image.width * resize, image.height * resize)
                    cached_images[key] = c

                return c

            info += f"Upscale: {round(upscaling_resize, 3)}, model:{shared.sd_upscalers[extras_upscaler_1].name}\n"
            res = upscale(image, extras_upscaler_1, upscaling_resize)

            if extras_upscaler_2 != 0 and extras_upscaler_2_visibility > 0:
                res2 = upscale(image, extras_upscaler_2, upscaling_resize)
                info += f"Upscale: {round(upscaling_resize, 3)}, visibility: {round(extras_upscaler_2_visibility, 3)}, model:{shared.sd_upscalers[extras_upscaler_2].name}\n"
                res = Image.blend(res, res2, extras_upscaler_2_visibility)

            image = res

        while len(cached_images) > 2:
            del cached_images[next(iter(cached_images.keys()))]

        images.save_image(image, path=outpath, basename="", seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,
                          no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo,
                          forced_filename=image_name if opts.use_original_name_batch else None)

        outputs.append(image)

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


def run_modelmerger(modelname_0, modelname_1, interp_method, interp_amount):
    # Linear interpolation (https://en.wikipedia.org/wiki/Linear_interpolation)
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    def sigmoid(theta0, theta1, alpha):
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    if os.path.exists(modelname_0):
        model0_filename = modelname_0
        modelname_0 = os.path.splitext(os.path.basename(modelname_0))[0]
    else:
        model0_filename = 'models/' + modelname_0 + '.ckpt'

    if os.path.exists(modelname_1):
        model1_filename = modelname_1
        modelname_1 = os.path.splitext(os.path.basename(modelname_1))[0]
    else:
        model1_filename = 'models/' + modelname_1 + '.ckpt'

    print(f"Loading {model0_filename}...")
    model_0 = torch.load(model0_filename, map_location='cpu')

    print(f"Loading {model1_filename}...")
    model_1 = torch.load(model1_filename, map_location='cpu')
    
    theta_0 = model_0['state_dict']
    theta_1 = model_1['state_dict']

    theta_funcs = {
        "Weighted Sum": weighted_sum,
        "Sigmoid": sigmoid,
    }
    theta_func = theta_funcs[interp_method]

    print(f"Merging...")
    for key in tqdm.tqdm(theta_0.keys()):
        if 'model' in key and key in theta_1:
            theta_0[key] = theta_func(theta_0[key], theta_1[key], interp_amount)
    
    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:
            theta_0[key] = theta_1[key]

    output_modelname = 'models/' + modelname_0 + '-' + modelname_1 + '-merged.ckpt'
    print(f"Saving to {output_modelname}...")
    torch.save(model_0, output_modelname)

    print(f"Checkpoint saved.")
    return "Checkpoint saved to " + output_modelname
