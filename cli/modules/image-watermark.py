#!/bin/env python
import os
import io
import pathlib
import argparse
import filetype
import numpy as np
from imwatermark import WatermarkEncoder, WatermarkDecoder
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from util import log, Map
import piexif
import piexif.helper


options = Map({ 'method': 'dwtDctSvd', 'type': 'bytes' })


def get_exif(image):
    # using piexif
    res1 = {}
    try:
        exif = piexif.load(image.info["exif"])
        exif = exif.get("Exif", {})
        for k, v in exif.items():
            key = list(vars(piexif.ExifIFD).keys())[list(vars(piexif.ExifIFD).values()).index(k)]
            res1[key] = piexif.helper.UserComment.load(v)
    except:
        pass
    # using pillow
    res2 = {}
    try:
        res2 = { TAGS[k]: v for k, v in image.getexif().items() if k in TAGS }
    except:
        pass
    return {**res1, **res2}


def set_exif(d: dict):
    ifd = ImageFileDirectory_v2()
    _TAGS = dict(((v, k) for k, v in TAGS.items())) # enumerate possible exif tags
    for k, v in d.items():
        ifd[_TAGS[k]] = v
    exif_stream = io.BytesIO()
    ifd.save(exif_stream)
    bytes = b'Exif\x00\x00' + exif_stream.getvalue()
    return bytes


def get_watermark(image, args):
    data = np.asarray(image)
    decoder = WatermarkDecoder(options.type, args.length)
    bytes = decoder.decode(data, options.method)
    try:
        watermark = str(bytes, 'UTF-8').replace('\x00', '')
    except:
        watermark = ''
    return watermark


def set_watermark(image, args):
    data = np.asarray(image)
    encoder = WatermarkEncoder()
    encoder.set_watermark(options.type, args.wm.encode('utf-8'))
    encoded = encoder.encode(data, options.method)
    image = Image.fromarray(encoded)
    return image


def watermark(args, file):
    if not os.path.exists(file):
        log.error({ 'watermark': 'file not found' })
        return
    if not filetype.is_image(file):
        log.error({ 'watermark': 'file is not an image' })
        return
    image = Image.open(file)
    if image.width * image.height < 256 * 256:
        log.error({ 'watermark': 'image too small' })
        return

    exif = get_exif(image)

    if args.command == 'read':
        watermark = get_watermark(image, args)
        log.info({ 'file': file, 'watermark': watermark, 'exif': exif, 'resolution': f'{image.width}x{image.height}' })

    elif args.command == 'write':
        metadata = b'' if args.strip else set_exif(exif)
        if args.output != '':
            pathlib.Path(args.output).mkdir(parents = True, exist_ok = True)
        image=set_watermark(image, args)
        fn = os.path.join(args.output, file)
        image.save(fn, exif=metadata)

        if args.verify:
            data = np.asarray(image)
            decoder = WatermarkDecoder(options.type, args.length)
            bytes = decoder.decode(data, options.method)
            if bytes.startswith(b'\xff'):
                watermark = ''
            else:
                watermark = str(bytes, 'UTF-8').replace('\x00', '')
        else:
            watermark = args.wm

        log.info({ 'file': fn, 'watermark': watermark, 'exif': None if args.strip else exif, 'resolution': f'{image.width}x{image.height}' })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'image watermarking')
    parser.add_argument('command', choices = ['read', 'write'])
    parser.add_argument('--wm', type=str, required=False, default='mm', help='watermark string')
    parser.add_argument('--strip', default=False, action='store_true', help = "strip existing exif data")
    parser.add_argument('--verify', default=False, action='store_true', help = "verify watermark during write")
    parser.add_argument('--length', type=int, default=16, help="watermark length in bits")
    parser.add_argument('--output', type=str, required=False, default='', help='folder to store images, default is overwrite in-place')
    parser.add_argument('input', type=str, nargs='*')
    args = parser.parse_args()
    log.info({ 'watermark args': vars(args), 'options': options })
    for arg in args.input:
        if os.path.isfile(arg):
            watermark(args, arg)
        elif os.path.isdir(arg):
            for root, _dirs, files in os.walk(arg):
                for f in files:
                    watermark(args, os.path.join(root, f))
