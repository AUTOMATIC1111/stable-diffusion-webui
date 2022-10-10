import datetime
import math
import os
from collections import namedtuple
import re

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from fonts.ttf import Roboto
import string

from modules import sd_samplers, shared
from modules.shared import opts, cmd_opts

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


def draw_grid_annotations(im, width, height, hor_texts, ver_texts):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2

    try:
        fnt = ImageFont.truetype(opts.font or Roboto, fontsize)
    except Exception:
        fnt = ImageFont.truetype(Roboto, fontsize)

    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else width * 3 // 4

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in
                        ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result


def draw_prompt_matrix(im, width, height, all_prompts):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)

    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]

    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    return draw_grid_annotations(im, width, height, hor_texts, ver_texts)


def resize_image(resize_mode, im, width, height):
    def resize(im, w, h):
        if opts.upscaler_for_img2img is None or opts.upscaler_for_img2img == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name == opts.upscaler_for_img2img]
            assert len(upscalers) > 0, f"could not find upscaler named {opts.upscaler_for_img2img}"

            upscaler = upscalers[0]
            im = upscaler.scaler.upscale(im, scale, upscaler.data_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


invalid_filename_chars = '<>:"/\\|?*\n'
invalid_filename_prefix = ' '
invalid_filename_postfix = ' .'
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
max_filename_part_length = 128


def sanitize_filename_part(text, replace_spaces=True):
    if replace_spaces:
        text = text.replace(' ', '_')

    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text


def apply_filename_pattern(x, p, seed, prompt):
    max_prompt_words = opts.directories_max_prompt_words

    if seed is not None:
        x = x.replace("[seed]", str(seed))

    if p is not None:
        x = x.replace("[steps]", str(p.steps))
        x = x.replace("[cfg]", str(p.cfg_scale))
        x = x.replace("[width]", str(p.width))
        x = x.replace("[height]", str(p.height))
        x = x.replace("[styles]", sanitize_filename_part(", ".join([x for x in p.styles if not x == "None"]) or "None", replace_spaces=False))
        x = x.replace("[sampler]", sanitize_filename_part(sd_samplers.samplers[p.sampler_index].name, replace_spaces=False))

    x = x.replace("[model_hash]", getattr(p, "sd_model_hash", shared.sd_model.sd_model_hash))
    x = x.replace("[date]", datetime.date.today().isoformat())
    x = x.replace("[datetime]", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    x = x.replace("[job_timestamp]", getattr(p, "job_timestamp", shared.state.job_timestamp))

    # Apply [prompt] at last. Because it may contain any replacement word.^M
    if prompt is not None:
        x = x.replace("[prompt]", sanitize_filename_part(prompt))
        if "[prompt_no_styles]" in x:
            prompt_no_style = prompt
            for style in shared.prompt_styles.get_style_prompts(p.styles):
                if len(style) > 0:
                    style_parts = [y for y in style.split("{prompt}")]
                    for part in style_parts:
                        prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",").strip().strip(',')
            prompt_no_style = prompt_no_style.replace(style, "").strip().strip(',').strip()
            x = x.replace("[prompt_no_styles]", sanitize_filename_part(prompt_no_style, replace_spaces=False))

        x = x.replace("[prompt_spaces]", sanitize_filename_part(prompt, replace_spaces=False))
        if "[prompt_words]" in x:
            words = [x for x in re_nonletters.split(prompt or "") if len(x) > 0]
            if len(words) == 0:
                words = ["empty"]
            x = x.replace("[prompt_words]", sanitize_filename_part(" ".join(words[0:max_prompt_words]), replace_spaces=False))

    if cmd_opts.hide_ui_dir_config:
        x = re.sub(r'^[\\/]+|\.{2,}[\\/]+|[\\/]+\.{2,}', '', x)

    return x


def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.

    The sequence starts at 0.
    """
    result = -1
    if basename != '':
        basename = basename + "-"

    prefix_length = len(basename)
    for p in os.listdir(path):
        if p.startswith(basename):
            l = os.path.splitext(p[prefix_length:])[0].split('-')  # splits the filename (removing the basename first if one is defined, so the sequence number is always the first element)
            try:
                result = max(int(l[0]), result)
            except ValueError:
                pass

    return result + 1


def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
    '''Save an image.

    Args:
        image (`PIL.Image`):
            The image to be saved.
        path (`str`):
            The directory to save the image. Note, the option `save_to_dirs` will make the image to be saved into a sub directory.
        basename (`str`):
            The base filename which will be applied to `filename pattern`.
        seed, prompt, short_filename, 
        extension (`str`):
            Image file extension, default is `png`.
        pngsectionname (`str`):
            Specify the name of the section which `info` will be saved in.
        info (`str` or `PngImagePlugin.iTXt`):
            PNG info chunks.
        existing_info (`dict`):
            Additional PNG info. `existing_info == {pngsectionname: info, ...}`
        no_prompt:
            TODO I don't know its meaning.
        p (`StableDiffusionProcessing`)
        forced_filename (`str`):
            If specified, `basename` and filename pattern will be ignored.
        save_to_dirs (bool):
            If true, the image will be saved into a subdirectory of `path`.

    Returns: (fullfn, txt_fullfn)
        fullfn (`str`):
            The full path of the saved imaged.
        txt_fullfn (`str` or None):
            If a text file is saved for this image, this will be its full path. Otherwise None.
    '''
    if short_filename or prompt is None or seed is None:
        file_decoration = ""
    elif opts.save_to_dirs:
        file_decoration = opts.samples_filename_pattern or "[seed]"
    else:
        file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

    if file_decoration != "":
        file_decoration = "-" + file_decoration.lower()

    file_decoration = apply_filename_pattern(file_decoration, p, seed, prompt) + suffix

    if extension == 'png' and opts.enable_pnginfo and info is not None:
        pnginfo = PngImagePlugin.PngInfo()

        if existing_info is not None:
            for k, v in existing_info.items():
                pnginfo.add_text(k, str(v))

        pnginfo.add_text(pnginfo_section_name, info)
    else:
        pnginfo = None

    if save_to_dirs is None:
        save_to_dirs = (grid and opts.grid_save_to_dirs) or (not grid and opts.save_to_dirs and not no_prompt)

    if save_to_dirs:
        dirname = apply_filename_pattern(opts.directories_filename_pattern or "[prompt_words]", p, seed, prompt).strip('\\ /')
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if forced_filename is None:
        basecount = get_next_sequence_number(path, basename)
        fullfn = "a.png"
        fullfn_without_extension = "a"
        for i in range(500):
            fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
            fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
            fullfn_without_extension = os.path.join(path, f"{fn}{file_decoration}")
            if not os.path.exists(fullfn):
                break
    else:
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")
        fullfn_without_extension = os.path.join(path, forced_filename)

    def exif_bytes():
        return piexif.dump({
            "Exif": {
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info or "", encoding="unicode")
            },
        })

    if extension.lower() in ("jpg", "jpeg", "webp"):
        image.save(fullfn, quality=opts.jpeg_quality)
        if opts.enable_pnginfo and info is not None:
            piexif.insert(exif_bytes(), fullfn)
    else:
        image.save(fullfn, quality=opts.jpeg_quality, pnginfo=pnginfo)

    target_side_length = 4000
    oversize = image.width > target_side_length or image.height > target_side_length
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > 4 * 1024 * 1024):
        ratio = image.width / image.height

        if oversize and ratio > 1:
            image = image.resize((target_side_length, image.height * target_side_length // image.width), LANCZOS)
        elif oversize:
            image = image.resize((image.width * target_side_length // image.height, target_side_length), LANCZOS)

        image.save(fullfn_without_extension + ".jpg", quality=opts.jpeg_quality)
        if opts.enable_pnginfo and info is not None:
            piexif.insert(exif_bytes(), fullfn_without_extension + ".jpg")

    if opts.save_txt and info is not None:
        txt_fullfn = f"{fullfn_without_extension}.txt"
        with open(txt_fullfn, "w", encoding="utf8") as file:
            file.write(info + "\n")
    else:
        txt_fullfn = None

    return fullfn, txt_fullfn
