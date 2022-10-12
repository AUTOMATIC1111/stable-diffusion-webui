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
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    if process_caption:
        shared.interrogator.load()

    if process_caption_deepbooru:
        deepbooru.create_deepbooru_process(opts.interrogate_deepbooru_score_threshold, opts.deepbooru_sort_alpha)

    def save_pic_with_caption(image, index):
        if process_caption:
            caption = "-" + shared.interrogator.generate_caption(image)
            caption = sanitize_caption(os.path.join(dst, f"{index:05}-{subindex[0]}"), caption, ".png")
        elif process_caption_deepbooru:
            shared.deepbooru_process_return["value"] = -1
            shared.deepbooru_process_queue.put(image)
            while shared.deepbooru_process_return["value"] == -1:
                time.sleep(0.2)
            caption = "-" + shared.deepbooru_process_return["value"]
            caption = sanitize_caption(os.path.join(dst, f"{index:05}-{subindex[0]}"), caption, ".png")
            shared.deepbooru_process_return["value"] = -1
        else:
            caption = filename
            caption = os.path.splitext(caption)[0]
            caption = os.path.basename(caption)

        image.save(os.path.join(dst, f"{index:05}-{subindex[0]}{caption}.png"))
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

    if process_caption:
        shared.interrogator.send_blip_to_ram()

    if process_caption_deepbooru:
        deepbooru.release_process()


def sanitize_caption(base_path, original_caption, suffix):
    operating_system = platform.system().lower()
    if (operating_system == "windows"):
        invalid_path_characters = "\\/:*?\"<>|"
        max_path_length = 259
    else:
        invalid_path_characters = "/" #linux/macos
        max_path_length = 1023
    caption = original_caption
    for invalid_character in invalid_path_characters:
        caption = caption.replace(invalid_character, "")
    fixed_path_length = len(base_path) + len(suffix) 
    if fixed_path_length + len(caption) <= max_path_length:
        return caption
    caption_tokens = caption.split()
    new_caption = ""
    for token in caption_tokens:
        last_caption = new_caption
        new_caption = new_caption + token + " "
        if (len(new_caption) + fixed_path_length - 1  > max_path_length):
            break
    print(f"\nPath will be too long. Truncated caption: {original_caption}\nto: {last_caption}", file=sys.stderr)
    return last_caption.strip()
