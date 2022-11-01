import os
from PIL import Image, ImageOps
import math
import platform
import sys
import tqdm
import time

from modules import shared, images
from modules.paths import models_path
from modules.shared import opts, cmd_opts
from modules.textual_inversion import autocrop
if cmd_opts.deepdanbooru:
    import modules.deepbooru as deepbooru


def preprocess(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9, process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False):
    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            db_opts = deepbooru.create_deepbooru_opts()
            db_opts[deepbooru.OPT_INCLUDE_RANKS] = False
            deepbooru.create_deepbooru_process(opts.interrogate_deepbooru_score_threshold, db_opts)

        preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru, split_threshold, overlap_ratio, process_focal_crop, process_focal_crop_face_weight, process_focal_crop_entropy_weight, process_focal_crop_edges_weight, process_focal_crop_debug)

    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.release_process()



def preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9, process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False):
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    def save_pic_with_caption(image, index, existing_caption=None):
        caption = ""

        if process_caption:
            caption += shared.interrogator.generate_caption(image)

        if process_caption_deepbooru:
            if len(caption) > 0:
                caption += ", "
            caption += deepbooru.get_tags_from_process(image)

        filename_part = filename
        filename_part = os.path.splitext(filename_part)[0]
        filename_part = os.path.basename(filename_part)

        basename = f"{index:05}-{subindex[0]}-{filename_part}"
        image.save(os.path.join(dst, f"{basename}.png"))

        if preprocess_txt_action == 'prepend' and existing_caption:
            caption = existing_caption + ' ' + caption
        elif preprocess_txt_action == 'append' and existing_caption:
            caption = caption + ' ' + existing_caption
        elif preprocess_txt_action == 'copy' and existing_caption:
            caption = existing_caption

        caption = caption.strip()
        
        if len(caption) > 0:
            with open(os.path.join(dst, f"{basename}.txt"), "w", encoding="utf8") as file:
                file.write(caption)

        subindex[0] += 1

    def save_pic(image, index, existing_caption=None):
        save_pic_with_caption(image, index, existing_caption=existing_caption)

        if process_flip:
            save_pic_with_caption(ImageOps.mirror(image), index, existing_caption=existing_caption)

    def split_pic(image, inverse_xy):
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


    for index, imagefile in enumerate(tqdm.tqdm(files)):
        subindex = [0]
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename).convert("RGB")
        except Exception:
            continue

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
            for splitted in split_pic(img, inverse_xy):
                save_pic(splitted, index, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                dnn_model_path = autocrop.download_and_cache_models(os.path.join(models_path, "opencv"))
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
                save_pic(focal, index, existing_caption=existing_caption)
            process_default_resize = False

        if process_default_resize:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index, existing_caption=existing_caption)

        shared.state.nextjob()