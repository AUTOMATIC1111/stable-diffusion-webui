from __future__ import annotations

import datetime
import math
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from fonts.ttf import Roboto
import string

from modules import sd_samplers, shared
from modules.shared import opts, cmd_opts

if TYPE_CHECKING:
    from modules.processing import Processed, StableDiffusionProcessing


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


INVALID_FILENAME_CHARS = '<>:"/\\|?*\n'
INVALID_FILENAME_CHAR_TABLE = {ord(x): "_" for x in INVALID_FILENAME_CHARS}
INVALID_FILENAME_PREFIX = ' '
INVALID_FILENAME_POSTFIX = ' .'

MAX_FILENAME_LENGTH = 240
MAX_FILENAME_PART_LENGTH = 128
TEXT_ELLIPSIS = "\u2026"
FILENAME_ELLIPSIS = "\u2026"

RE_NONLETTERS = re.compile(r'[\s' + string.punctuation + ']+')


def split_prompt_words(prompt: str) -> list[str]:
    """Split prompt into words"""
    return [x for x in RE_NONLETTERS.split(prompt) if len(x) > 0]


def truncate_text(text: str, length: int, ellipsis=TEXT_ELLIPSIS) -> str:
    """Truncate text to specified length and append ellipsis

    Examples::
    >>> truncate_filename("foobar", length=6)
    'foobar'
    >>> truncate_filename("foobar", length=4)
    'foo…'
    >>> truncate_filename("foobarbaz", length=8, ellipsis="(trunc)")
    'f(trunc)'
    """
    if length <= 0:
        raise ValueError("length must be 1 or more")
    if length < len(ellipsis):
        return ellipsis[:length]
    if len(text) > length:
        return text[:length - len(ellipsis)] + ellipsis
    return text


def sanitize_pathname(pathname: str, length=MAX_FILENAME_LENGTH):
    """Sanitize invalid pathname and truncate length"""
    path = Path(pathname)
    if path.drive or path.root:
        # disallow absolute path
        parts = path.parts[1:] 
    else:
        parts = path.parts

    res: list[str] = []

    for part in parts:
        part = part.translate(INVALID_FILENAME_CHAR_TABLE)
        part = part.lstrip(INVALID_FILENAME_PREFIX)

        # truncate only basename
        basename, extension = os.path.splitext(part)
        basename = basename.strip()
        if length > 0:
            trunclen = max(1, length - len(extension))
            basename = truncate_text(basename, length=trunclen, ellipsis=FILENAME_ELLIPSIS)
        part = basename + extension

        part = part.rstrip(INVALID_FILENAME_POSTFIX)

        # ignore empty or `.` only parts
        if part.strip("."):
            res.append(part)

    return str(Path(*res)) if res else ""


def sanitize_filename_part(text: str, replace_spaces=True, length=MAX_FILENAME_PART_LENGTH):
    """Sanitize invalid chars in part of filename and truncate length"""
    if replace_spaces:
        text = text.replace(" ", "_")
    text = text.strip().translate(INVALID_FILENAME_CHAR_TABLE)
    if length > 0:
        text = truncate_text(text, length, ellipsis=FILENAME_ELLIPSIS)
    return text


def apply_filename_pattern(
    x: str,
    p: StableDiffusionProcessing | Processed | None = None,
    seed: int | None = None,
    prompt: str | None = None,
    index=0,
) -> str:
    def replace_pattern(keyword: str, value_or_func):
        nonlocal x
        pattern = f"[{keyword}]"
        if pattern in x:
            value = value_or_func() if callable(value_or_func) else value_or_func
            x = x.replace(pattern, f"{value}")

    if p:
        replace_pattern("steps", p.steps)
        replace_pattern("cfg", p.cfg_scale)
        replace_pattern("width", p.width)
        replace_pattern("height", p.height)
        replace_pattern("styles", lambda: sanitize_filename_part(", ".join(x for x in p.styles if x != "None") or "No styles", replace_spaces=False))
        replace_pattern("sampler", lambda: sanitize_filename_part(sd_samplers.samplers[p.sampler_index].name))
        replace_pattern("model_hash", lambda: getattr(p, "sd_model_hash", shared.sd_model.sd_model_hash))
        replace_pattern("job_timestamp", lambda: getattr(p, "job_timestamp", shared.state.job_timestamp))
    else:
        replace_pattern("model_hash", shared.sd_model.sd_model_hash)
        replace_pattern("job_timestamp", shared.state.job_timestamp)

    if seed is not None:
        replace_pattern("seed", seed)

    replace_pattern("date", lambda: datetime.date.today().isoformat())
    replace_pattern("datetime", lambda: datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    # Apply [prompt] at last. Because it may contain any replacement word.
    if prompt is not None:
        replace_pattern("prompt", lambda: sanitize_filename_part(prompt))
        replace_pattern("prompt_spaces", lambda: sanitize_filename_part(prompt, replace_spaces=False))
        replace_pattern("prompt_words", lambda: sanitize_filename_part(" ".join(split_prompt_words(prompt)[:opts.directories_max_prompt_words]) or "empty", replace_spaces=False))
    if p:
        replace_pattern("prompt_no_styles", lambda: sanitize_filename_part(p.prompt[index - p.index_of_first_image] if isinstance(p.prompt, list) else p.prompt, replace_spaces=False))

    if cmd_opts.hide_ui_dir_config:
        x = re.sub(r'^[\\/]+|\.{2,}[\\/]+|[\\/]+\.{2,}', '', x)

    return x


def get_next_sequence_number(path: str, basename: str, separator="-"):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.

    The sequence starts at 0.
    """
    if basename:
        basename += separator

    prefix_length = len(basename)
    sequences = (
        # get the sequence number (removing the basename and the extension,
        # so the sequence number is always the first element)
        os.path.splitext(p[prefix_length:])[0].split(separator, 1)[0]
        for p in os.listdir(path)
        if p.startswith(basename)
    )

    try:
        return max(int(s) for s in sequences if s.isdigit()) + 1
    except ValueError: # when sequences is empty
        return 0


def save_image(
    image: Image.Image,
    path: str,
    basename: str,
    seed: int | None = None,
    prompt: str | None = None,
    extension="png",
    info: str | None = None,
    short_filename=False,
    no_prompt=False,
    grid=False,
    pnginfo_section_name="parameters",
    p: StableDiffusionProcessing | Processed | None = None,
    existing_info: dict | None = None,
    forced_filename: str | None = None,
    index=0,
    save_to_dirs: bool | None = None,
    separator="-",
) -> str:
    if save_to_dirs is None:
        save_to_dirs = opts.grid_save_to_dirs if grid else (opts.save_to_dirs and not no_prompt)

    if save_to_dirs:
        dirname_pattern = (opts.directories_filename_pattern or "[prompt_words]").lower()
        dirname = apply_filename_pattern(dirname_pattern, p, seed, prompt, index)
        dirname = sanitize_pathname(dirname)
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if forced_filename is None:
        if short_filename or prompt is None or seed is None:
            filename_pattern = ""
        elif opts.samples_filename_pattern:
            filename_pattern = opts.samples_filename_pattern.lower()
        elif save_to_dirs:
            filename_pattern = "[seed]"
        else:
            filename_pattern = "[seed]-[prompt_spaces]"

        file_decoration = apply_filename_pattern(filename_pattern, p, seed, prompt, index)
        sequence_number = get_next_sequence_number(path, basename, separator)
        fn = separator.join(filter(None, [
            basename,
            f"{sequence_number:04}" if basename else f"{sequence_number:05}",
            file_decoration,
        ]))
        fn = sanitize_pathname(f"{fn}.{extension}")
    else:
        fn = f"{forced_filename}.{extension}"
    fullfn = os.path.join(path, fn)
    fullfn_without_extension = fullfn[:len(fullfn) - len(extension) - 1]

    if opts.enable_pnginfo and info:
        pnginfo = PngImagePlugin.PngInfo()
        if existing_info is not None:
            for k, v in existing_info.items():
                pnginfo.add_text(k, f"{v}")
        pnginfo.add_text(pnginfo_section_name, info)

        exif_bytes = piexif.dump({
            "Exif": {
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info, encoding="unicode")
            },
        })
    else:
        pnginfo = None
        exif_bytes = None

    image.save(fullfn, quality=opts.jpeg_quality, pnginfo=pnginfo)
    if extension.lower() in ("jpg", "jpeg", "webp") and  exif_bytes:
        piexif.insert(exif_bytes, fullfn)

    MAX_FILE_SIZE_FOR_4CHAN = 4 * 1024 * 1024
    TARGET_SIDE_LENGTH = 4000
    oversize = image.width > TARGET_SIDE_LENGTH or image.height > TARGET_SIDE_LENGTH
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > MAX_FILE_SIZE_FOR_4CHAN):
        if oversize:
            ratio = image.width / image.height
            if ratio > 1:
                w = TARGET_SIDE_LENGTH
                h = image.height * TARGET_SIDE_LENGTH // image.width
            else:
                w = image.width * TARGET_SIDE_LENGTH // image.height
                h = TARGET_SIDE_LENGTH
            image = image.resize((w, h), LANCZOS)

        fullfn_jpg = f"{fullfn_without_extension}.jpg"
        image.save(fullfn_jpg, quality=opts.jpeg_quality)
        if exif_bytes:
            piexif.insert(exif_bytes, fullfn_jpg)

    if opts.save_txt and info:
        fullfn_txt = f"{fullfn_without_extension}.txt"
        with open(fullfn_txt, "w", encoding="utf8") as file:
            file.write(f"{info}\n")

    return fullfn
