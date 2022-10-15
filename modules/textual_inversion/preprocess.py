import os
from PIL import Image, ImageOps
import platform
import sys
import tqdm
import time

from modules import shared, images
from modules.shared import opts, cmd_opts
if cmd_opts.deepdanbooru:
    import modules.deepbooru as deepbooru


def preprocess(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru=False):
    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            db_opts = deepbooru.create_deepbooru_opts()
            db_opts[deepbooru.OPT_INCLUDE_RANKS] = False
            deepbooru.create_deepbooru_process(opts.interrogate_deepbooru_score_threshold, db_opts)

        preprocess_work(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru)

    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.release_process()



def preprocess_work(process_src, process_dst, process_width, process_height, process_flip, process_split, process_caption, process_caption_deepbooru=False):
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

        if process_split and is_tall:
            img = img.resize((width, height * img.height // img.width))

            top = img.crop((0, 0, width, height))
            save_pic(top, index)

            bot = img.crop((0, img.height - height, width, img.height))
            save_pic(bot, index)
        elif process_split and is_wide:
            img = img.resize((width * img.width // img.height, height))

            left = img.crop((0, 0, width, height))
            save_pic(left, index)

            right = img.crop((img.width - width, 0, img.width, height))
            save_pic(right, index)
        else:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index)

        shared.state.nextjob()
