import datetime
import io
import re
import os
import math
import json
import string
import hashlib
import queue
import threading
from collections import namedtuple
import pytz
import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin, ExifTags
from modules import sd_samplers, shared, script_callbacks, errors, paths

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


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
        elif shared.opts.grid_prevent_empty_spots:
            rows = math.floor(math.sqrt(len(imgs)))
            while len(imgs) % rows != 0:
                rows -= 1
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)
    if rows > len(imgs):
        rows = len(imgs)
    cols = math.ceil(len(imgs) / rows)
    params = script_callbacks.ImageGridLoopParams(imgs, cols, rows)
    script_callbacks.image_grid_callback(params)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(params.cols * w, params.rows * h), color='black')
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


def draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin=0):
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
            return ImageFont.truetype(shared.opts.font or 'javascript/roboto.ttf', fontsize)
        except Exception:
            return ImageFont.truetype('javascript/roboto.ttf', fontsize)

    def draw_texts(drawing, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        for _i, line in enumerate(lines):
            fnt = initial_fnt
            fontsize = initial_fontsize
            while drawing.multiline_textsize(line.text, font=fnt)[0] > line.allowed_width and fontsize > 0:
                fontsize -= 1
                fnt = get_font(fontsize)
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")
            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)
            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = get_font(fontsize)
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
            line.allowed_width = allowed_width
    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]
    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2
    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + margin * (rows-1)), "white")
    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + (height + margin) * row))
    d = ImageDraw.Draw(result)
    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2
        draw_texts(d, x, y, hor_texts[col], fnt, fontsize)
    for row in range(rows):
        x = pad_left / 2
        y = pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2
        draw_texts(d, x, y, ver_texts[row], fnt, fontsize)
    return result


def draw_prompt_matrix(im, width, height, all_prompts, margin=0):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]
    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]
    return draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin)


def resize_image(resize_mode, im, width, height, upscaler_name=None):
    """
    Resizes an image with the specified resize_mode, width, and height.
    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    upscaler_name = upscaler_name or shared.opts.upscaler_for_img2img

    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)
        scale = max(w / im.width, h / im.height)
        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            if len(upscalers) == 0:
                upscaler = shared.sd_upscalers[0]
                shared.log.warning(f"Could not find upscaler: {upscaler_name or '<empty string>'} using fallback: {upscaler.name}")
            else:
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
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
max_filename_part_length = 128
NOTHING_AND_SKIP_PREVIOUS_TEXT = object()


def sanitize_filename_part(text, replace_spaces=True):
    if text is None:
        return None
    text = os.path.basename(text)
    if replace_spaces:
        text = text.replace(' ', '_')
    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text


class FilenameGenerator:
    replacements = {
        'seed': lambda self: self.seed if self.seed is not None else '',
        'steps': lambda self: self.p and self.p.steps,
        'cfg': lambda self: self.p and self.p.cfg_scale,
        'width': lambda self: self.image.width,
        'height': lambda self: self.image.height,
        'styles': lambda self: self.p and sanitize_filename_part(", ".join([style for style in self.p.styles if not style == "None"]) or "None", replace_spaces=False),
        'sampler': lambda self: self.p and sanitize_filename_part(self.p.sampler_name, replace_spaces=False),
        'model_hash': lambda self: getattr(self.p, "sd_model_hash", shared.sd_model.sd_model_hash),
        'model_name': lambda self: sanitize_filename_part(shared.sd_model.sd_checkpoint_info.model_name, replace_spaces=False),
        'model_shortname': lambda self: sanitize_filename_part(shared.sd_model.sd_checkpoint_info.name_for_extra, replace_spaces=False),
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'prompt_hash': lambda self: hashlib.sha256(self.prompt.encode()).hexdigest()[0:8],
        'prompt': lambda self: sanitize_filename_part(self.prompt),
        'prompt_no_styles': lambda self: self.prompt_no_style(),
        'prompt_spaces': lambda self: sanitize_filename_part(self.prompt, replace_spaces=False),
        'prompt_words': lambda self: self.prompt_words(),
        'batch_number': lambda self: NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.batch_size == 1 else self.p.batch_index + 1,
        'generation_number': lambda self: NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.n_iter == 1 and self.p.batch_size == 1 else self.p.iteration * self.p.batch_size + self.p.batch_index + 1,
        'hasprompt': lambda self, *args: self.hasprompt(*args),  # accepts formats:[hasprompt<prompt1|default><prompt2>..]
        'clip_skip': lambda self: self.p and self.p.clip_skip,
        'denoising': lambda self: self.p.denoising_strength if self.p and self.p.denoising_strength else NOTHING_AND_SKIP_PREVIOUS_TEXT,
    }
    default_time_format = '%Y%m%d%H%M%S'

    def __init__(self, p, seed, prompt, image):
        self.p = p
        self.seed = seed
        self.prompt = prompt
        self.image = image

    def hasprompt(self, *args):
        lower = self.prompt.lower()
        if self.p is None or self.prompt is None:
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
        return sanitize_filename_part(outres)

    def prompt_no_style(self):
        if self.p is None or self.prompt is None:
            return None
        prompt_no_style = self.prompt
        for style in shared.prompt_styles.get_style_prompts(self.p.styles):
            if len(style) > 0:
                for part in style.split("{prompt}"):
                    prompt_no_style = prompt_no_style.replace(part, "").replace(", ,", ",").strip().strip(',')

                prompt_no_style = prompt_no_style.replace(style, "").strip().strip(',').strip()
        return sanitize_filename_part(prompt_no_style, replace_spaces=False)

    def prompt_words(self):
        words = [x for x in re_nonletters.split(self.prompt or "") if len(x) > 0]
        if len(words) == 0:
            words = ["empty"]
        return sanitize_filename_part(" ".join(words[0:shared.opts.directories_max_prompt_words]), replace_spaces=False)

    def datetime(self, *args):
        time_datetime = datetime.datetime.now()
        time_format = args[0] if len(args) > 0 and args[0] != "" else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError as _:
            time_zone = None
        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError) as _:
            formatted_time = time_zone_time.strftime(self.default_time_format)
        return sanitize_filename_part(formatted_time, replace_spaces=False)

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
            fun = self.replacements.get(pattern.lower())
            if fun is not None:
                try:
                    replacement = fun(self, *pattern_args)
                except Exception as e:
                    replacement = None
                    errors.display(e, 'filename pattern')
                if replacement == NOTHING_AND_SKIP_PREVIOUS_TEXT:
                    continue
                elif replacement is not None:
                    res += text + str(replacement)
                    continue
            res += f'{text}[{pattern}]'
            res = res.split('?')[0]
        return res


def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.
    """
    result = -1
    if basename != '':
        basename = f"{basename}-"
    prefix_length = len(basename)
    for p in os.listdir(path):
        if p.startswith(basename):
            l = os.path.splitext(p[prefix_length:])[0].split('-')  # splits the filename (removing the basename first if one is defined, so the sequence number is always the first element)
            try:
                result = max(int(l[0]), result)
            except ValueError:
                pass
    return result + 1


def atomically_save_image():
    Image.MAX_IMAGE_PIXELS = None # disable check in Pillow and rely on check below to allow large custom image sizes
    while True:
        image, filename, extension, params, exifinfo_data, txt_fullfn = save_queue.get()
        fn = filename + extension
        image_format = Image.registered_extensions()[extension]
        shared.log.debug(f'Saving image: {image_format} {fn} {image.size}')
        # actual save
        if image_format == 'PNG':
            pnginfo_data = PngImagePlugin.PngInfo()
            for k, v in params.pnginfo.items():
                pnginfo_data.add_text(k, str(v))
            image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, pnginfo=pnginfo_data)
        elif image_format == 'JPEG':
            if image.mode == 'RGBA':
                shared.log.warning('Saving RGBA image as JPEG: Alpha channel will be lost')
                image = image.convert("RGB")
            elif image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("L")
            exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo_data or "", encoding="unicode") } })
            image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, exif=exif_bytes)
        elif image_format == 'WEBP':
            if image.mode == 'I;16':
                image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
            exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(exifinfo_data or "", encoding="unicode") } })
            image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
        else:
            shared.log.warning(f'Unrecognized image format: {extension} attempting save as {image_format}')
            image.save(fn, format=image_format, quality=shared.opts.jpeg_quality)
        # additional metadata saved in files
        if shared.opts.save_txt and len(exifinfo_data) > 0:
            with open(txt_fullfn, "w", encoding="utf8") as file:
                file.write(f"{exifinfo_data}\n")
        with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            file.write(exifinfo_data)
        if shared.opts.save_log_fn != '' and len(exifinfo_data) > 0:
            try:
                with open(os.path.join(paths.data_path, shared.opts.save_log_fn), mode='a+', encoding='utf-8') as f:
                    try:
                        entries = json.load(f)
                    except:
                        entries = []
                    f.seek(0)
                    entries.append({ 'filename': filename, 'time': datetime.datetime.now().isoformat(), 'info': exifinfo_data })
                    json.dump(entries, f, indent=4)
                    del entries
            except Exception as e:
                shared.log.warning(f'Failed to save log file: {shared.opts.save_log_fn} {e}')
        save_queue.task_done()


save_queue = queue.Queue()
save_thread = threading.Thread(target=atomically_save_image, daemon=True)
save_thread.start()


def save_image(image, path, basename, seed=None, prompt=None, extension='jpg', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
    """Save an image.
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
    """
    if image is None:
        shared.log.warning('Image is none')
        return None, None
    if not check_grid_size([image]):
        return None, None
    if path is None: # set default path to avoid errors when functions are triggered manually or via api and param is not set
        path = shared.opts.outdir_save
    namegen = FilenameGenerator(p, seed, prompt, image)
    if save_to_dirs is None:
        save_to_dirs = (grid and shared.opts.grid_save_to_dirs) or (not grid and shared.opts.save_to_dirs and not no_prompt)
    if save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)
    os.makedirs(path, exist_ok=True)
    if forced_filename is None:
        if short_filename or seed is None:
            file_decoration = ""
        if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0:
            file_decoration = shared.opts.samples_filename_pattern
        else:
            file_decoration = "[seed]-[prompt_spaces]"
        add_number = shared.opts.save_images_add_number or file_decoration == ''
        if file_decoration != "" and add_number:
            file_decoration = f"-{file_decoration}"
        file_decoration = namegen.apply(file_decoration) + suffix
        if add_number:
            basecount = get_next_sequence_number(path, basename)
            fullfn = None
            for i in range(500):
                fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                if not os.path.exists(fullfn):
                    break
        else:
            fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    else:
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")
    pnginfo = existing_info or {}
    if info is not None:
        pnginfo[pnginfo_section_name] = info
    params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    script_callbacks.before_image_saved_callback(params)
    exifinfo_data = params.pnginfo.get('UserComment', '')
    if len(exifinfo_data) > 0:
        exifinfo_data = exifinfo_data + ', ' + params.pnginfo.get(pnginfo_section_name, '')
    else:
        exifinfo_data = params.pnginfo.get(pnginfo_section_name, '')
    filename, extension = os.path.splitext(params.filename)
    if hasattr(os, 'statvfs'):
        max_name_len = os.statvfs(path).f_namemax
        filename = filename[:max_name_len - max(4, len(extension))]
        params.filename = filename + extension
    txt_fullfn = f"{filename}.txt" if shared.opts.save_txt and len(exifinfo_data) > 0 else None

    save_queue.put((params.image, filename, extension, params, exifinfo_data, txt_fullfn))
    save_queue.join()
    # atomically_save_image(params.image, filename, extension, params, exifinfo_data, txt_fullfn)

    params.image.already_saved_as = params.filename
    script_callbacks.image_saved_callback(params)
    return params.filename, txt_fullfn


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
        except:
            pass
    return None


def read_info_from_image(image):
    items = image.info or {}
    geninfo = items.pop('parameters', None)
    if geninfo is None:
        geninfo = items.pop('UserComment', None)
    if geninfo is not None and len(geninfo) > 0:
        if 'UserComment' in geninfo:
            geninfo = geninfo['UserComment']
        items['UserComment'] = geninfo

    if "exif" in items:
        exif = piexif.load(items["exif"])
        for _key, subkey in exif.items():
            if isinstance(subkey, dict):
                for key, val in subkey.items():
                    if isinstance(val, bytes): # decode bytestring
                        val = safe_decode_string(val)
                    if isinstance(val, tuple) and isinstance(val[0], int) and isinstance(val[1], int): # convert camera ratios
                        val = round(val[0] / val[1], 2)
                    if val is not None and key in ExifTags.TAGS: # add known tags
                        if ExifTags.TAGS[key] == 'UserComment': # add geninfo from UserComment
                            geninfo = val
                            items['parameters'] = val
                        else:
                            items[ExifTags.TAGS[key]] = val
                    elif val is not None and key in ExifTags.GPSTAGS:
                        items[ExifTags.GPSTAGS[key]] = val

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
