import os
from PIL import Image, ImageOps
import tqdm

from modules import shared, images


def preprocess(process_src, process_dst, process_flip, process_split, process_caption):
    size = 512
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)

    assert src != dst, 'same directory specified as source and desitnation'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    if process_caption:
        shared.interrogator.load()

    def save_pic_with_caption(image, index):
        if process_caption:
            caption = "-" + shared.interrogator.generate_caption(image)
        else:
            caption = ""

        image.save(os.path.join(dst, f"{index:05}-{subindex[0]}{caption}.png"))
        subindex[0] += 1

    def save_pic(image, index):
        save_pic_with_caption(image, index)

        if process_flip:
            save_pic_with_caption(ImageOps.mirror(image), index)

    for index, imagefile in enumerate(tqdm.tqdm(files)):
        subindex = [0]
        filename = os.path.join(src, imagefile)
        img = Image.open(filename).convert("RGB")

        if shared.state.interrupted:
            break

        ratio = img.height / img.width
        is_tall = ratio > 1.35
        is_wide = ratio < 1 / 1.35

        if process_split and is_tall:
            img = img.resize((size, size * img.height // img.width))

            top = img.crop((0, 0, size, size))
            save_pic(top, index)

            bot = img.crop((0, img.height - size, size, img.height))
            save_pic(bot, index)
        elif process_split and is_wide:
            img = img.resize((size * img.width // img.height, size))

            left = img.crop((0, 0, size, size))
            save_pic(left, index)

            right = img.crop((img.width - size, 0, img.width, size))
            save_pic(right, index)
        else:
            img = images.resize_image(1, img, size, size)
            save_pic(img, index)

        shared.state.nextjob()

    if process_caption:
        shared.interrogator.send_blip_to_ram()
