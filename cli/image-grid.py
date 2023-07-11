#!/usr/bin/env python
"""
Create image grid
"""

import os
import argparse
import math
import logging
from pathlib import Path
import filetype
from PIL import Image, ImageDraw, ImageFont
from util import log


params = None


def wrap(text: str, font: ImageFont.ImageFont, length: int):
    lines = ['']
    for word in text.split():
        line = f'{lines[-1]} {word}'.strip()
        if font.getlength(line) <= length:
            lines[-1] = line
        else:
            lines.append(word)
    return '\n'.join(lines)


def grid(images, labels = None, width = 0, height = 0, border = 0, square = False, horizontal = False, vertical = False): # pylint: disable=redefined-outer-name
    if horizontal:
        rows = 1
    elif vertical:
        rows = len(images)
    elif square:
        rows = round(math.sqrt(len(images)))
    else:
        rows = math.floor(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    size = [0, 0]
    if width == 0:
        w = max([i.size[0] for i in images])
        size[0] = cols * w + cols * border
    else:
        size[0] = width
        w = round(width / cols)
    if height == 0:
        h = max([i.size[1] for i in images])
        size[1] = rows * h + rows * border
    else:
        size[1] = height
        h = round(height / rows)
    size = tuple(size)
    image = Image.new('RGB', size = size, color = 'black') # pylint: disable=redefined-outer-name
    font = ImageFont.truetype('DejaVuSansMono', round(w / 20))
    for i, img in enumerate(images): # pylint: disable=redefined-outer-name
        x = (i % cols * w) + (i % cols * border)
        y = (i // cols * h) + (i // cols * border)
        img.thumbnail((w, h), Image.HAMMING)
        image.paste(img, box=(x, y))
        if labels is not None and len(images) == len(labels):
            ctx = ImageDraw.Draw(image)
            label = wrap(labels[i], font, w)
            ctx.text((x + 1 + round(w / 200), y + 1 + round(w / 200)), label, font = font, fill = (0, 0, 0))
            ctx.text((x, y), label, font = font, fill = (255, 255, 255))
    log.info({ 'grid': { 'images': len(images), 'rows': rows, 'cols': cols, 'cell': [w, h] } })
    return image


if __name__ == '__main__':
    log.info({ 'create grid' })
    parser = argparse.ArgumentParser(description='image grid utility')
    parser.add_argument("--square", default = False, action='store_true', help = "create square grid")
    parser.add_argument("--horizontal", default = False, action='store_true', help = "create horizontal grid")
    parser.add_argument("--vertical", default = False, action='store_true', help = "create vertical grid")
    parser.add_argument("--width", type = int, default = 0, required = False, help = "fixed grid width")
    parser.add_argument("--height", type = int, default = 0, required = False, help = "fixed grid height")
    parser.add_argument("--border", type = int, default = 0, required = False, help = "image border")
    parser.add_argument('--nolabels', default = False, action='store_true', help = "do not print image labels")
    parser.add_argument('--debug', default = False, action='store_true', help = "print extra debug information")
    parser.add_argument('output', type = str)
    parser.add_argument('input', type = str, nargs = '*')
    params = parser.parse_args()
    output = params.output if params.output.lower().endswith('.jpg') else params.output + '.jpg'
    if params.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': params.__dict__ })
    images = []
    labels = []
    for f in params.input:
        path = Path(f)
        if path.is_dir():
            files = [os.path.join(f, file) for file in os.listdir(f) if os.path.isfile(os.path.join(f, file))]
        elif path.is_file():
            files = [f]
        else:
            log.warning({ 'grid not a valid file/folder', f})
            continue
        files.sort()
        for file in files:
            if not filetype.is_image(file):
                continue
            if file.lower().endswith('.heic'):
                from pi_heif import register_heif_opener
                register_heif_opener()
            log.debug(file)
            img = Image.open(file)
            # img.verify()
            images.append(img)
            fp = Path(file)
            if not params.nolabels:
                labels.append(fp.stem)
    # log.info({ 'folder': path.parent, 'labels': labels })
    if len(images) > 0:
        image = grid(
            images = images,
            labels = labels,
            width = params.width,
            height = params.height,
            border = params.border,
            square = params.square,
            horizontal = params.horizontal,
            vertical = params.vertical)
        image.save(output, 'JPEG', optimize = True, quality = 60)
        log.info({ 'grid': { 'file': output, 'size': list(image.size) } })
    else:
        log.info({ 'grid': 'nothing to do' })
