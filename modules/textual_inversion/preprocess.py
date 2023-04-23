import os
from PIL import Image, ImageOps
import math
import platform
import sys
import tqdm
import time

from modules import paths, shared, images, deepbooru
from modules.shared import opts, cmd_opts
from modules.textual_inversion import autocrop


def preprocess(id_task, process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9, process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None, process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None, process_multicrop_objective=None, process_multicrop_threshold=None, process_rotate=None, process_rotate_angle=None, process_rotate_steps=None):
    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            deepbooru.model.start()

        preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru, split_threshold, overlap_ratio, process_focal_crop, process_focal_crop_face_weight, process_focal_crop_entropy_weight, process_focal_crop_edges_weight, process_focal_crop_debug, process_multicrop, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea, process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold, process_rotate, process_rotate_angle, process_rotate_steps)

    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.model.stop()


def listfiles(dirname):
    return os.listdir(dirname)


class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    rotate = False
    rotate_angle = 0.0
    rotate_steps = 0
    process_caption = False
    process_caption_deepbooru = False
    preprocess_txt_action = None


def save_pic_with_caption(image, index, params: PreprocessParams, existing_caption=None):
    caption = ""

    if params.process_caption:
        caption += shared.interrogator.generate_caption(image)

    if params.process_caption_deepbooru:
        if len(caption) > 0:
            caption += ", "
        caption += deepbooru.model.tag_multi(image)

    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.preprocess_txt_action == 'prepend' and existing_caption:
        caption = existing_caption + ' ' + caption
    elif params.preprocess_txt_action == 'append' and existing_caption:
        caption = caption + ' ' + existing_caption
    elif params.preprocess_txt_action == 'copy' and existing_caption:
        caption = existing_caption

    caption = caption.strip()

    if len(caption) > 0:
        with open(os.path.join(params.dstdir, f"{basename}.txt"), "w", encoding="utf8") as file:
            file.write(caption)

    params.subindex += 1


def save_pic(image, index, params, existing_caption=None, rotate=True):
    save_pic_with_caption(image, index, params, existing_caption=existing_caption)

    if params.rotate and rotate:
        for i in range(params.rotate_steps):
            save_pic(rotate_crop(image, params.rotate_angle / params.rotate_steps * (i + 1)), index, params, existing_caption=existing_caption, rotate=False)

    if params.flip:
        save_pic_with_caption(ImageOps.mirror(image), index, params, existing_caption=existing_caption)


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted

# not using torchvision.transforms.CenterCrop because it doesn't allow float regions
def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1-(lambda x: x if x < 1 else 1/x)(iw/ih/(w/h))
    wh = max(((w, h) for w in range(mindim, maxdim+1, 64) for h in range(mindim, maxdim+1, 64)
        if minarea <= w * h <= maxarea and err(w, h) <= threshold),
        key= lambda wh: (wh[0]*wh[1], -err(*wh))[::1 if objective=='Maximize area' else -1],
        default=None
    )
    return wh and center_crop(image, *wh)
    

def rotate_crop(image, angle):
    rotated = image.rotate(angle, Image.BICUBIC, True)

    # Calculate the aspect ratios and rotation angle in radians
    aspect_ratio = float(image.size[0]) / image.size[1]
    rotated_aspect_ratio = float(rotated.size[0]) / rotated.size[1]
    angle = math.fabs(angle) * math.pi / 180

    # Calculate the total height of the rotated image
    if aspect_ratio < 1:
        total_height = float(image.size[0]) / rotated_aspect_ratio
    else:
        total_height = float(image.size[1])

    # Calculate the height and width of the output image after rotation
    h = total_height / (aspect_ratio * math.fabs(math.sin(angle)) + math.fabs(math.cos(angle)))
    w = h * aspect_ratio

    # Calculate the cropping coordinates
    left = (rotated.size[0] - w) / 2
    top = (rotated.size[1] - h) / 2
    right = (rotated.size[0] + w) / 2
    bottom = (rotated.size[1] + h) / 2

    # Crop the rotated image
    cropped = rotated.crop((left, top, right, bottom))
    return cropped

def preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9, process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None, process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None, process_multicrop_objective=None, process_multicrop_threshold=None, process_rotate=None, process_rotate_angle=None, process_rotate_steps=None):
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = listfiles(src)

    shared.state.job = "preprocess"
    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    params = PreprocessParams()
    params.dstdir = dst
    params.flip = process_flip
    params.rotate = process_rotate
    params.rotate_angle = process_rotate_angle
    params.rotate_steps = int(process_rotate_steps)
    params.process_caption = process_caption
    params.process_caption_deepbooru = process_caption_deepbooru
    params.preprocess_txt_action = preprocess_txt_action

    pbar = tqdm.tqdm(files)
    for index, imagefile in enumerate(pbar):
        params.subindex = 0
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename).convert("RGB")
        except Exception:
            continue

        description = f"Preprocessing [Image {index}/{len(files)}]"
        pbar.set_description(description)
        shared.state.textinfo = description

        params.src = filename

        existing_caption = None
        existing_caption_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(existing_caption_filename):
            with open(existing_caption_filename, 'r', encoding="utf8") as file:
                existing_caption = file.read()

        if shared.state.interrupted:
            break

        if img.height > img.width:
            ratio = (img.width * height) / (img.height * width)
            inverse_xy = False
        else:
            ratio = (img.height * width) / (img.width * height)
            inverse_xy = True

        process_default_resize = True

        if process_split and ratio < 1.0 and ratio <= split_threshold:
            for splitted in split_pic(img, inverse_xy, width, height, overlap_ratio):
                save_pic(splitted, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                dnn_model_path = autocrop.download_and_cache_models(os.path.join(paths.models_path, "opencv"))
            except Exception as e:
                print("Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.", e)

            autocrop_settings = autocrop.Settings(
                crop_width = width,
                crop_height = height,
                face_points_weight = process_focal_crop_face_weight,
                entropy_points_weight = process_focal_crop_entropy_weight,
                corner_points_weight = process_focal_crop_edges_weight,
                annotate_image = process_focal_crop_debug,
                dnn_model_path = dnn_model_path,
            )
            for focal in autocrop.crop_image(img, autocrop_settings):
                save_pic(focal, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_multicrop:
            cropped = multicrop_pic(img, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea, process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)
            if cropped is not None:
                save_pic(cropped, index, params, existing_caption=existing_caption)
            else:
                print(f"skipped {img.width}x{img.height} image {filename} (can't find suitable size within error threshold)")
            process_default_resize = False

        if process_default_resize:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index, params, existing_caption=existing_caption)

        shared.state.nextjob()
