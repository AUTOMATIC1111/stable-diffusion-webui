import io
import re
import os
import sys
import math
import json
import uuid
import queue
import string
import hashlib
import datetime
import threading
from pathlib import Path
from collections import namedtuple
import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin, ExifTags
from modules import sd_samplers, shared, script_callbacks, errors, paths


debug = errors.log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None
try:
    from pi_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


def check_grid_size(imgs):
    mp = 0
    for img in imgs:
        mp += img.width * img.height
    mp = round(mp / 1000000)
    ok = mp <= shared.opts.img_max_size_mp
    if not ok:
        shared.log.warning(f'Maximum image size exceded: size={mp} maximum={shared.opts.img_max_size_mp} MPixels')
    return ok


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if shared.opts.n_rows > 0:
            rows = shared.opts.n_rows
        elif shared.opts.n_rows == 0:
            rows = batch_size
        else:
            rows = math.floor(math.sqrt(len(imgs)))
            while len(imgs) % rows != 0:
                rows -= 1
    if rows > len(imgs):
        rows = len(imgs)
    cols = math.ceil(len(imgs) / rows)
    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(params.cols * w, params.rows * h), color=shared.opts.grid_background)
    for i, img in enumerate(params.imgs):
        grid.paste(img, box=(i % params.cols * w, i // params.cols * h))
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


def draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin=0, title=None):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def get_font(fontsize):
        try:
            return ImageFont.truetype(shared.opts.font or 'javascript/notosans-nerdfont-regular.ttf', fontsize)
        except Exception:
            return ImageFont.truetype('javascript/notosans-nerdfont-regular.ttf', fontsize)

    def draw_texts(drawing: ImageDraw, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        for line in lines:
            font = initial_fnt
            fontsize = initial_fontsize
            while drawing.multiline_textbbox((0,0), text=line.text, font=font)[0] > line.allowed_width and fontsize > 0:
                fontsize -= 1
                font = get_font(fontsize)
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=font, fill=shared.opts.font_color if line.is_active else color_inactive, anchor="mm", align="center")
            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)
            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    font = get_font(fontsize)
    color_inactive = (127, 127, 127)
    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else width * 3 // 4
    cols = im.width // width
    rows = im.height // height
    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'
    calc_img = Image.new("RGB", (1, 1), shared.opts.grid_background)
    calc_d = ImageDraw.Draw(calc_img)
    title_texts = [title] if title else [[GridAnnotation()]]
    for texts, allowed_width in zip(hor_texts + ver_texts + title_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts) + [(width+margin)*cols]):
        items = [] + texts
        texts.clear()
        for line in items:
            wrapped = wrap(calc_d, line.text, font, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]
        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=font)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            line.allowed_width = allowed_width
    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]
    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2
    title_pad = 0
    if title:
        title_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in title_texts] # pylint: disable=unsubscriptable-object
        title_pad = 0 if sum(title_text_heights) == 0 else max(title_text_heights) + line_spacing * 2
    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + title_pad + margin * (rows-1)), shared.opts.grid_background)
    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + title_pad + (height + margin) * row))
    d = ImageDraw.Draw(result)
    if title:
        x = pad_left + ((width+margin)*cols) / 2
        y = title_pad / 2 - title_text_heights[0] / 2
        draw_texts(d, x, y, title_texts[0], font, fontsize)
    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = (pad_top / 2 - hor_text_heights[col] / 2) + title_pad
        draw_texts(d, x, y, hor_texts[col], font, fontsize)
    for row in range(rows):
        x = pad_left / 2
        y = (pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2) + title_pad
        draw_texts(d, x, y, ver_texts[row], font, fontsize)
    return result


def draw_prompt_matrix(im, width, height, all_prompts, margin=0):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]
    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]
    return draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin)


def resize_image(resize_mode, im, width, height, upscaler_name=None, output_type='image'):
    shared.log.debug(f'Image resize: mode={resize_mode} resolution={width}x{height} upscaler={upscaler_name} function={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access
    """
    Resizes an image with the specified resize_mode, width, and height.
    Args:
        resize_mode: The mode to use when resizing the image.
            0: No resize
            1: Resize the image to the specified width and height.
            2: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            3: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    upscaler_name = upscaler_name or shared.opts.upscaler_for_img2img

    def latent(im, w, h, upscaler):
        from modules.processing_vae import vae_encode, vae_decode
        import torch
        latents = vae_encode(im, shared.sd_model, full_quality=False) # TODO enable full VAE mode
        latents = torch.nn.functional.interpolate(latents, size=(h // 8, w // 8), mode=upscaler["mode"], antialias=upscaler["antialias"])
        im = vae_decode(latents, shared.sd_model, output_type='pil', full_quality=False)[0]
        return im

    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=Image.Resampling.LANCZOS)
        scale = max(w / im.width, h / im.height)
        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            if len(upscalers) > 0:
                upscaler = upscalers[0]
                im = upscaler.scaler.upscale(im, scale, upscaler.data_path)
            else:
                upscaler = shared.latent_upscale_modes.get(upscaler_name, None)
                if upscaler is not None:
                    im = latent(im, w, h, upscaler)
                else:
                    shared.log.warning(f"Could not find upscaler: {upscaler_name or '<empty string>'} using fallback: {upscaler.name}")
        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=Image.Resampling.LANCZOS)
        return im

    if resize_mode == 0 or (im.width == width and im.height == height):
        res = im.copy()
    elif resize_mode == 1:
        res = resize(im, width, height)
    elif resize_mode == 2:
        ratio = width / height
        src_ratio = im.width / im.height
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height
        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))
    if output_type == 'np':
        return np.array(res)
    return res


re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
re_attention = re.compile(r'[\(*\[*](\w+)(:\d+(\.\d+))?[\)*\]*]|')
re_network = re.compile(r'\<\w+:(\w+)(:\d+(\.\d+))?\>|')
re_brackets = re.compile(r'[\([{})\]]')

NOTHING = object()


class FilenameGenerator:
    replacements = {
        'width': lambda self: self.image.width,
        'height': lambda self: self.image.height,
        'batch_number': lambda self: self.batch_number,
        'iter_number': lambda self: self.iter_number,
        'num': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'generation_number': lambda self: NOTHING if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        'hasprompt': lambda self, *args: self.hasprompt(*args),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        'hash': lambda self: self.image_hash(),
        'image_hash': lambda self: self.image_hash(),
        'timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),

        'model': lambda self: shared.sd_model.sd_checkpoint_info.title,
        'model_shortname': lambda self: shared.sd_model.sd_checkpoint_info.model_name,
        'model_name': lambda self: shared.sd_model.sd_checkpoint_info.model_name,
        'model_hash': lambda self: shared.sd_model.sd_checkpoint_info.shorthash,

        'prompt': lambda self: self.prompt_full(),
        'prompt_no_styles': lambda self: self.prompt_no_style(),
        'prompt_words': lambda self: self.prompt_words(),
        'prompt_hash': lambda self: hashlib.sha256(self.prompt.encode()).hexdigest()[0:8],

        'sampler': lambda self: self.p and self.p.sampler_name,
        'seed': lambda self: self.seed and str(self.seed) or '',
        'steps': lambda self: self.p and self.p.steps,
        'styles': lambda self: self.p and ", ".join([style for style in self.p.styles if not style == "None"]) or "None",
        'uuid': lambda self: str(uuid.uuid4()),
    }
    default_time_format = '%Y%m%d%H%M%S'

    def __init__(self, p, seed, prompt, image, grid=False):
        if p is None:
            debug('Filename generator init skip')
        else:
            debug(f'Filename generator init: {seed} {prompt}')
        self.p = p
        if seed is not None and seed > 0:
            self.seed = seed
        elif hasattr(p, 'all_seeds'):
            self.seed = p.all_seeds[0]
        else:
            self.seed = 0
        self.prompt = prompt
        self.image = image
        if not grid:
            self.batch_number = NOTHING if self.p is None or getattr(self.p, 'batch_size', 1) == 1 else (self.p.batch_index + 1 if hasattr(self.p, 'batch_index') else NOTHING)
            self.iter_number = NOTHING if self.p is None or getattr(self.p, 'n_iter', 1) == 1 else (self.p.iteration + 1 if hasattr(self.p, 'iteration') else NOTHING)
        else:
            self.batch_number = NOTHING
            self.iter_number = NOTHING

    def hasprompt(self, *args):
        lower = self.prompt.lower()
        if getattr(self, 'p', None) is None or getattr(self, 'prompt', None) is None:
            return None
        outres = ""
        for arg in args:
            if arg != "":
                division = arg.split("|")
                expected = division[0].lower()
                default = division[1] if len(division) > 1 else ""
                if lower.find(expected) >= 0:
                    outres = f'{outres}{expected}'
                else:
                    outres = outres if default == "" else f'{outres}{default}'
        return outres

    def image_hash(self):
        if getattr(self, 'image', None) is None:
            return None
        import base64
        from io import BytesIO
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        shorthash = hashlib.sha256(img_str).hexdigest()[0:8]
        return shorthash

    def prompt_full(self):
        return self.prompt_sanitize(self.prompt)

    def prompt_words(self):
        if getattr(self, 'prompt', None) is None:
            return ''
        no_attention = re_attention.sub(r'\1', self.prompt)
        no_network = re_network.sub(r'\1', no_attention)
        no_brackets = re_brackets.sub('', no_network)
        words = [x for x in re_nonletters.split(no_brackets or "") if len(x) > 0]
        prompt = " ".join(words[0:shared.opts.directories_max_prompt_words])
        return self.prompt_sanitize(prompt)

    def prompt_no_style(self):
        if getattr(self, 'p', None) is None or getattr(self, 'prompt', None) is None:
            return None
        prompt_no_style = self.prompt
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if len(style) > 0:
                for part in style.split("{prompt}"):
                    prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",")
                prompt_no_style = prompt_no_style.replace(style, "")
        return self.prompt_sanitize(prompt_no_style)

    def datetime(self, *args):
        import pytz
        time_datetime = datetime.datetime.now()
        time_format = args[0] if len(args) > 0 and args[0] != "" else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError:
            time_zone = None
        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError):
            formatted_time = time_zone_time.strftime(self.default_time_format)
        return formatted_time

    def prompt_sanitize(self, prompt):
        invalid_chars = '#<>:\'"\\|?*\n\t\r'
        sanitized = prompt.translate({ ord(x): '_' for x in invalid_chars }).strip()
        debug(f'Prompt sanitize: input="{prompt}" output={sanitized}')
        return sanitized

    def sanitize(self, filename):
        invalid_chars = '\'"|?*\n\t\r' # <https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file>
        invalid_folder = ':'
        invalid_files = ['CON', 'PRN', 'AUX', 'NUL', 'NULL', 'COM0', 'COM1', 'LPT0', 'LPT1']
        invalid_prefix = ', '
        invalid_suffix = '.,_ '
        fn, ext = os.path.splitext(filename)
        parts = Path(fn).parts
        newparts = []
        for i, part in enumerate(parts):
            part = part.translate({ ord(x): '_' for x in invalid_chars })
            if i > 0 or (len(part) >= 2 and part[1] != invalid_folder): # skip drive, otherwise remove
                part = part.translate({ ord(x): '_' for x in invalid_folder })
            part = part.lstrip(invalid_prefix).rstrip(invalid_suffix)
            if part in invalid_files: # reserved names
                [part := part.replace(word, '_') for word in invalid_files] # pylint: disable=expression-not-assigned
            newparts.append(part)
        fn = str(Path(*newparts))
        max_length = max(256 - len(ext), os.statvfs(__file__).f_namemax - 32 if hasattr(os, 'statvfs') else 256 - len(ext))
        while len(os.path.abspath(fn)) > max_length:
            fn = fn[:-1]
        fn += ext
        debug(f'Filename sanitize: input="{filename}" parts={parts} output="{fn}" ext={ext} max={max_length} len={len(fn)}')
        return fn

    def sequence(self, x, dirname, basename):
        if shared.opts.save_images_add_number or '[seq]' in x:
            if '[seq]' not in x:
                x = os.path.join(os.path.dirname(x), f"[seq]-{os.path.basename(x)}")
            basecount = get_next_sequence_number(dirname, basename)
            for i in range(9999):
                seq = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                filename = x.replace('[seq]', seq)
                if not os.path.exists(filename):
                    debug(f'Prompt sequence: input="{x}" seq={seq} output="{filename}"')
                    x = filename
                    break
        return x

    def apply(self, x):
        res = ''
        for m in re_pattern.finditer(x):
            text, pattern = m.groups()
            if pattern is None:
                res += text
                continue
            pattern_args = []
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break
                pattern, arg = m.groups()
                pattern_args.insert(0, arg)
            fun = self.replacements.get(pattern.lower(), None)
            if fun is not None:
                try:
                    debug(f'Filename apply: pattern={pattern.lower()} args={pattern_args}')
                    replacement = fun(self, *pattern_args)
                except Exception as e:
                    replacement = None
                    shared.log.error(f'Filename apply pattern: {x} {e}')
                if replacement == NOTHING:
                    continue
                if replacement is not None:
                    res += text + str(replacement).replace('/', '-').replace('\\', '-')
                    continue
            else:
                res += text + f'[{pattern}]' # reinsert unknown pattern
        return res


def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.
    """
    result = -1
    if basename != '':
        basename = f"{basename}-"
    prefix_length = len(basename)
    if not os.path.isdir(path):
        return 0
    for p in os.listdir(path):
        if p.startswith(basename):
            parts = os.path.splitext(p[prefix_length:])[0].split('-')  # splits the filename (removing the basename first if one is defined, so the sequence number is always the first element)
            try:
                result = max(int(parts[0]), result)
            except ValueError:
                pass
    return result + 1


def atomically_save_image():
    Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes
    while True:
        image, filename, extension, params, exifinfo, filename_txt = save_queue.get()
        with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            file.write(exifinfo)
        fn = filename + extension
        filename = filename.strip()
        if extension[0] != '.': # add dot if missing
            extension = '.' + extension
        try:
            image_format = Image.registered_extensions()[extension]
        except Exception:
            shared.log.warning(f'Unknown image format: {extension}')
            image_format = 'JPEG'
        if shared.opts.image_watermark_enabled:
            image = set_watermark(image, shared.opts.image_watermark)
        size = os.path.getsize(fn) if os.path.exists(fn) else 0
        shared.log.debug(f'Saving: image="{fn}" type={image_format} resolution={image.width}x{image.height} size={size}')
        # additional metadata saved in files
        if shared.opts.save_txt and len(exifinfo) > 0:
            try:
                with open(filename_txt, "w", encoding="utf8") as file:
                    file.write(f"{exifinfo}\n")
                shared.log.debug(f'Saving: text="{filename_txt}" len={len(exifinfo)}')
            except Exception as e:
                shared.log.warning(f'Image description save failed: {filename_txt} {e}')
        # actual save
        exifinfo = (exifinfo or "") if shared.opts.image_metadata else ""
        if image_format == 'PNG':
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in params.pnginfo.items():
                pnginfo_data.add_text(k, str(v))
            try:
                image.save(fn, format=image_format, compress_level=6, pnginfo=pnginfo_data if shared.opts.image_metadata else None)
            except Exception as e:
                shared.log.error(f'Image save failed: file="{fn}" {e}')
        elif image_format == 'JPEG':
            if image.mode == 'RGBA':
                shared.log.warning('Saving RGBA image as JPEG: Alpha channel will be lost')
                image = image.convert("RGB")
            elif image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("L")
            exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
            try:
                image.save(fn, format=image_format, optimize=True, quality=shared.opts.jpeg_quality, exif=exif_bytes)
            except Exception as e:
                shared.log.error(f'Image save failed: file="{fn}" {e}')
        elif image_format == 'WEBP':
            if image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
            exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo, encoding="unicode") } })
            try:
                image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
            except Exception as e:
                shared.log.error(f'Image save failed: file="{fn}" {e}')
        else:
            # shared.log.warning(f'Unrecognized image format: {extension} attempting save as {image_format}')
            try:
                image.save(fn, format=image_format, quality=shared.opts.jpeg_quality)
            except Exception as e:
                shared.log.error(f'Image save failed: file="{fn}" {e}')
        if shared.opts.save_log_fn != '' and len(exifinfo) > 0:
            fn = os.path.join(paths.data_path, shared.opts.save_log_fn)
            if not fn.endswith('.json'):
                fn += '.json'
            entries = shared.readfile(fn, silent=True)
            idx = len(list(entries))
            if idx == 0:
                entries = []
            entry = { 'id': idx, 'filename': filename, 'time': datetime.datetime.now().isoformat(), 'info': exifinfo }
            entries.append(entry)
            shared.writefile(entries, fn, mode='w', silent=True)
            shared.log.debug(f'Saving: json="{fn}" records={len(entries)}')
        save_queue.task_done()


save_queue = queue.Queue()
save_thread = threading.Thread(target=atomically_save_image, daemon=True)
save_thread.start()


def save_image(image, path, basename='', seed=None, prompt=None, extension=shared.opts.samples_format, info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix='', save_to_dirs=None): # pylint: disable=unused-argument
    debug(f'Save from function={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access
    if image is None:
        shared.log.warning('Image is none')
        return None, None
    if not check_grid_size([image]):
        return None, None
    if path is None or len(path) == 0: # set default path to avoid errors when functions are triggered manually or via api and param is not set
        path = shared.opts.outdir_save
    namegen = FilenameGenerator(p, seed, prompt, image, grid=grid)
    suffix = suffix if suffix is not None else ''
    basename = basename if basename is not None else ''
    if shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        path = os.path.join(path, dirname)
    if forced_filename is None:
        if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0:
            file_decoration = shared.opts.samples_filename_pattern
        else:
            file_decoration = "[seq]-[prompt_words]"
        file_decoration = namegen.apply(file_decoration)
        file_decoration += suffix if suffix is not None else ''
        filename = os.path.join(path, f"{file_decoration}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{file_decoration}.{extension}")
    else:
        forced_filename += suffix if suffix is not None else ''
        filename = os.path.join(path, f"{forced_filename}.{extension}") if basename == '' else os.path.join(path, f"{basename}-{forced_filename}.{extension}")
    pnginfo = existing_info or {}
    if info is not None:
        pnginfo[pnginfo_section_name] = info
    params = script_callbacks.ImageSaveParams(image, p, filename, pnginfo)
    params.filename = namegen.sanitize(filename)
    dirname = os.path.dirname(params.filename)
    if dirname is not None and len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    params.filename = namegen.sequence(params.filename, dirname, basename)
    params.filename = namegen.sanitize(params.filename)
    # callbacks
    script_callbacks.before_image_saved_callback(params)
    exifinfo = params.pnginfo.get('UserComment', '')
    exifinfo = (exifinfo + ', ' if len(exifinfo) > 0 else '') + params.pnginfo.get(pnginfo_section_name, '')
    filename, extension = os.path.splitext(params.filename)
    filename_txt = f"{filename}.txt" if shared.opts.save_txt and len(exifinfo) > 0 else None
    save_queue.put((params.image, filename, extension, params, exifinfo, filename_txt)) # actual save is executed in a thread that polls data from queue
    save_queue.join()
    if not hasattr(params.image, 'already_saved_as'):
        debug(f'Image marked: "{params.filename}"')
        params.image.already_saved_as = params.filename
    script_callbacks.image_saved_callback(params)
    return params.filename, filename_txt


def save_video_atomic(images, filename, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3):
    try:
        import cv2
    except Exception as e:
        shared.log.error(f'Save video: cv2: {e}')
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if video_type.lower() == 'mp4':
        frames = images
        if interpolate > 0:
            try:
                import modules.rife
                frames = modules.rife.interpolate(images, count=interpolate, scale=scale, pad=pad, change=change)
            except Exception as e:
                shared.log.error(f'RIFE interpolation: {e}')
                errors.display(e, 'RIFE interpolation')
        video_frames = [np.array(frame) for frame in frames]
        fourcc = "mp4v"
        h, w, _c = video_frames[0].shape
        video_writer = cv2.VideoWriter(filename, fourcc=cv2.VideoWriter_fourcc(*fourcc), fps=len(frames)/duration, frameSize=(w, h))
        for i in range(len(video_frames)):
            img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        size = os.path.getsize(filename)
        shared.log.info(f'Save video: file="{filename}" frames={len(frames)} duration={duration} fourcc={fourcc} size={size}')
    if video_type.lower() == 'gif' or video_type.lower() == 'png':
        append = images.copy()
        image = append.pop(0)
        if loop:
            append += append[::-1]
        frames=len(append) + 1
        image.save(
            filename,
            save_all = True,
            append_images = append,
            optimize = False,
            duration = 1000.0 * duration / frames,
            loop = 0 if loop else 1,
        )
        size = os.path.getsize(filename)
        shared.log.info(f'Save video: file="{filename}" frames={len(append) + 1} duration={duration} loop={loop} size={size}')


def save_video(p, images, filename = None, video_type: str = 'none', duration: float = 2.0, loop: bool = False, interpolate: int = 0, scale: float = 1.0, pad: int = 1, change: float = 0.3, sync: bool = False):
    if images is None or len(images) < 2 or video_type is None or video_type.lower() == 'none':
        return
    image = images[0]
    if p is not None:
        namegen = FilenameGenerator(p, seed=p.all_seeds[0], prompt=p.all_prompts[0], image=image)
    else:
        namegen = FilenameGenerator(None, seed=0, prompt='', image=image)
    if filename is None and p is not None:
        filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
        filename = os.path.join(shared.opts.outdir_video, filename)
        filename = namegen.sequence(filename, shared.opts.outdir_video, '')
    else:
        if os.pathsep not in filename:
            filename = os.path.join(shared.opts.outdir_video, filename)
    if not filename.lower().endswith(video_type.lower()):
        filename += f'.{video_type.lower()}'
    filename = namegen.sanitize(filename)
    if not sync:
        threading.Thread(target=save_video_atomic, args=(images, filename, video_type, duration, loop, interpolate, scale, pad, change)).start()
    else:
        save_video_atomic(images, filename, video_type, duration, loop, interpolate, scale, pad, change)
    return filename


def safe_decode_string(s: bytes):
    remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text # pylint: disable=unnecessary-lambda-assignment
    for encoding in ['utf-8', 'utf-16', 'ascii', 'latin_1', 'cp1252', 'cp437']: # try different encodings
        try:
            s = remove_prefix(s, b'UNICODE')
            s = remove_prefix(s, b'ASCII')
            s = remove_prefix(s, b'\x00')
            val = s.decode(encoding, errors="strict")
            val = re.sub(r'[\x00-\x09]', '', val).strip() # remove remaining special characters
            if len(val) == 0: # remove empty strings
                val = None
            return val
        except Exception:
            pass
    return None


def read_info_from_image(image: Image):
    items = image.info or {}
    geninfo = items.pop('parameters', None)
    if geninfo is None:
        geninfo = items.pop('UserComment', None)
    if geninfo is not None and len(geninfo) > 0:
        if 'UserComment' in geninfo:
            geninfo = geninfo['UserComment']
        items['UserComment'] = geninfo

    if "exif" in items:
        try:
            exif = piexif.load(items["exif"])
        except Exception as e:
            shared.log.error(f'Error loading EXIF data: {e}')
            exif = {}
        for _key, subkey in exif.items():
            if isinstance(subkey, dict):
                for key, val in subkey.items():
                    if isinstance(val, bytes): # decode bytestring
                        val = safe_decode_string(val)
                    if isinstance(val, tuple) and isinstance(val[0], int) and isinstance(val[1], int) and val[1] > 0: # convert camera ratios
                        val = round(val[0] / val[1], 2)
                    if val is not None and key in ExifTags.TAGS: # add known tags
                        if ExifTags.TAGS[key] == 'UserComment': # add geninfo from UserComment
                            geninfo = val
                            items['parameters'] = val
                        else:
                            items[ExifTags.TAGS[key]] = val
                    elif val is not None and key in ExifTags.GPSTAGS:
                        items[ExifTags.GPSTAGS[key]] = val
    wm = get_watermark(image)
    if wm != '':
        # geninfo += f' Watermark: {wm}'
        items['watermark'] = wm

    for key, val in items.items():
        if isinstance(val, bytes): # decode bytestring
            items[key] = safe_decode_string(val)

    for key in ['exif', 'ExifOffset', 'JpegIFOffset', 'JpegIFByteCount', 'ExifVersion', 'icc_profile', 'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'adobe', 'photoshop', 'loop', 'duration', 'dpi']: # remove unwanted tags
        items.pop(key, None)

    if items.get("Software", None) == "NovelAI":
        try:
            json_info = json.loads(items["Comment"])
            sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")
            geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
        except Exception as e:
            errors.display(e, 'novelai image parser')

    try:
        items['width'] = image.width
        items['height'] = image.height
        items['mode'] = image.mode
    except Exception:
        pass

    return geninfo, items


def image_data(data):
    import gradio as gr
    if data is None:
        return gr.update(), None
    err1 = None
    err2 = None
    try:
        image = Image.open(io.BytesIO(data))
        errors.log.debug(f'Decoded object: image={image}')
        textinfo, _ = read_info_from_image(image)
        return textinfo, None
    except Exception as e:
        err1 = e
    try:
        if len(data) > 1024 * 10:
            errors.log.warning(f'Error decoding object: data too long: {len(data)}')
            return gr.update(), None
        text = data.decode('utf8')
        errors.log.debug(f'Decoded object: size={len(text)}')
        return text, None
    except Exception as e:
        err2 = e
    errors.log.error(f'Error decoding object: {err1 or err2}')
    return gr.update(), None


def flatten(img, bgcolor):
    """replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency"""
    if img.mode == "RGBA":
        background = Image.new('RGBA', img.size, bgcolor)
        background.paste(img, mask=img)
        img = background
    return img.convert('RGB')


def set_watermark(image, watermark):
    from imwatermark import WatermarkEncoder
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = 32
    length = wm_length // 8
    info = image.info
    data = np.asarray(image)
    encoder = WatermarkEncoder()
    text = f"{watermark:<{length}}"[:length]
    bytearr = text.encode(encoding='ascii', errors='ignore')
    try:
        encoder.set_watermark(wm_type, bytearr)
        encoded = encoder.encode(data, wm_method)
        image = Image.fromarray(encoded)
        image.info = info
        shared.log.debug(f'Set watermark: {watermark} method={wm_method} bits={wm_length}')
    except Exception as e:
        shared.log.warning(f'Set watermark error: {watermark} method={wm_method} bits={wm_length} {e}')
    return image


def get_watermark(image):
    from imwatermark import WatermarkDecoder
    wm_type = 'bytes'
    wm_method = 'dwtDctSvd'
    wm_length = 32
    data = np.asarray(image)
    decoder = WatermarkDecoder(wm_type, wm_length)
    try:
        decoded = decoder.decode(data, wm_method)
        wm = decoded.decode(encoding='ascii', errors='ignore')
    except Exception:
        wm = ''
    return wm
