import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import platform
import sys
import tqdm
import time

from modules import shared, images
from modules.shared import opts, cmd_opts
if cmd_opts.deepdanbooru:
    import modules.deepbooru as deepbooru


def preprocess(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru=False, process_entropy_focus=False):
    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            db_opts = deepbooru.create_deepbooru_opts()
            db_opts[deepbooru.OPT_INCLUDE_RANKS] = False
            deepbooru.create_deepbooru_process(opts.interrogate_deepbooru_score_threshold, db_opts)

        preprocess_work(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru, process_entropy_focus)

    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.release_process()



def preprocess_work(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru=False, process_entropy_focus=False):
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    def save_pic_with_caption(image, index):
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

        if len(caption) > 0:
            with open(os.path.join(dst, f"{basename}.txt"), "w", encoding="utf8") as file:
                file.write(caption)

        subindex[0] += 1

    def save_pic(image, index):
        save_pic_with_caption(image, index)

        if process_flip:
            save_pic_with_caption(ImageOps.mirror(image), index)

    for index, imagefile in enumerate(tqdm.tqdm(files)):
        subindex = [0]
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename).convert("RGB")
        except Exception:
            continue

        if shared.state.interrupted:
            break

        ratio = img.height / img.width
        is_tall = ratio > 1.35
        is_wide = ratio < 1 / 1.35

        processing_option_ran = False

        if process_split and is_tall:
            img = img.resize((width, height * img.height // img.width))

            top = img.crop((0, 0, width, height))
            save_pic(top, index)

            bot = img.crop((0, img.height - height, width, img.height))
            save_pic(bot, index)

            processing_option_ran = True
        elif process_split and is_wide:
            img = img.resize((width * img.width // img.height, height))

            left = img.crop((0, 0, width, height))
            save_pic(left, index)

            right = img.crop((img.width - width, 0, img.width, height))
            save_pic(right, index)
            
            processing_option_ran = True

        if process_entropy_focus and (is_tall or is_wide):
            if is_tall:
                img = img.resize((width, height * img.height // img.width))
            else:
                img = img.resize((width * img.width // img.height, height))

            x_focal_center, y_focal_center = image_central_focal_point(img, width, height)

            # take the focal point and turn it into crop coordinates that try to center over the focal
            # point but then get adjusted back into the frame
            y_half = int(height / 2)
            x_half = int(width / 2)

            x1 = x_focal_center - x_half
            if x1 < 0:
                x1 = 0
            elif x1 + width > img.width:
                x1 = img.width - width

            y1 = y_focal_center - y_half
            if y1 < 0:
                y1 = 0
            elif y1 + height > img.height:
                y1 = img.height - height

            x2 = x1 + width
            y2 = y1 + height

            crop = [x1, y1, x2, y2]

            focal = img.crop(tuple(crop))
            save_pic(focal, index)

            processing_option_ran = True

        if not processing_option_ran:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index)

        shared.state.nextjob()


def image_central_focal_point(im, target_width, target_height):
    focal_points = []

    focal_points.extend(
        image_focal_points(im)
    )

    fp_entropy = image_entropy_point(im, target_width, target_height)
    fp_entropy['weight'] = len(focal_points) + 1 # about half of the weight to entropy

    focal_points.append(fp_entropy)

    weight = 0.0
    x = 0.0
    y = 0.0
    for focal_point in focal_points:
        weight += focal_point['weight']
        x += focal_point['x'] * focal_point['weight']
        y += focal_point['y'] * focal_point['weight']
    avg_x = round(x // weight)
    avg_y = round(y // weight)

    return avg_x, avg_y


def image_focal_points(im):
    grayscale = im.convert("L")

    # naive attempt at preventing focal points from collecting at watermarks near the bottom
    gd = ImageDraw.Draw(grayscale)
    gd.rectangle([0, im.height*.9, im.width, im.height], fill="#999")

    np_im = np.array(grayscale)

    points = cv2.goodFeaturesToTrack(
        np_im,
        maxCorners=100,
        qualityLevel=0.04,
        minDistance=min(grayscale.width, grayscale.height)*0.07,
        useHarrisDetector=False,
    )

    if points is None:
        return []

    focal_points = []
    for point in points:
        x, y = point.ravel()
        focal_points.append({
            'x': x,
            'y': y,
            'weight': 1.0
        })

    return focal_points


def image_entropy_point(im, crop_width, crop_height):
    landscape = im.height < im.width
    portrait = im.height > im.width
    if landscape:
      move_idx = [0, 2]
      move_max = im.size[0]
    elif portrait:
      move_idx = [1, 3]
      move_max = im.size[1]

    e_max = 0
    crop_current = [0, 0, crop_width, crop_height]
    crop_best = crop_current
    while crop_current[move_idx[1]] < move_max:
        crop = im.crop(tuple(crop_current))
        e = image_entropy(crop)

        if (e > e_max):
          e_max = e
          crop_best = list(crop_current)

        crop_current[move_idx[0]] += 4
        crop_current[move_idx[1]] += 4

    x_mid = int(crop_best[0] + crop_width/2)
    y_mid = int(crop_best[1] + crop_height/2)


    return {
        'x': x_mid,
        'y': y_mid,
        'weight': 1.0
    }


def image_entropy(im):
    # greyscale image entropy
    band = np.asarray(im.convert("1"))
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()

