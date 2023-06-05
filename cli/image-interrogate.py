#!/usr/bin/env python
"""
use clip to interrogate image(s)
"""

import io
import base64
import sys
import os
import asyncio
import filetype
from PIL import Image
from util import log, Map
import sdapi


stats = { 'captions': {}, 'keywords': {} }
exclude = ['a', 'in', 'on', 'out', 'at', 'the', 'and', 'with', 'next', 'to', 'it', 'for', 'of', 'into', 'that']


def decode(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


def encode(f):
    image = Image.open(f)
    exif = image.getexif()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG', exif = exif)
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def print_summary():
    captions = dict(sorted(stats['captions'].items(), key=lambda x:x[1], reverse=True))
    log.info({ 'caption stats': captions })
    keywords = dict(sorted(stats['keywords'].items(), key=lambda x:x[1], reverse=True))
    log.info({ 'keyword stats': keywords })


async def interrogate(f):
    if not filetype.is_image(f):
        log.info({ 'interrogate skip': f })
        return
    json = Map({ 'image': encode(f) })
    log.info({ 'interrogate': f })
    # run clip
    json.model = 'clip'
    res = await sdapi.post('/sdapi/v1/interrogate', json)
    caption = ""
    style = ""
    if 'caption' in res:
        caption = res.caption
        log.info({ 'interrogate caption': caption })
        if ', by' in caption:
            style = caption.split(', by')[1].strip()
            log.info({ 'interrogate style': style })
        for word in caption.split(' '):
            if word not in exclude:
                stats['captions'][word] = stats['captions'][word] + 1 if word in stats['captions'] else 1
    else:
        log.error({ 'interrogate clip error': res })
    # run booru
    json.model = 'deepdanbooru'
    res = await sdapi.post('/sdapi/v1/interrogate', json)
    keywords = {}
    if 'caption' in res:
        for term in res.caption.split(', '):
            term = term.replace('(', '').replace(')', '').replace('\\', '').split(':')
            if len(term) < 2:
                continue
            keywords[term[0]] = term[1]
        keywords = dict(sorted(keywords.items(), key=lambda x:x[1], reverse=True))
        for word in keywords.items():
            stats['keywords'][word[0]] = stats['keywords'][word[0]] + 1 if word[0] in stats['keywords'] else 1
        log.info({ 'interrogate keywords': keywords })
    else:
        log.error({ 'interrogate booru error': res })
    return caption, keywords, style


async def main():
    sys.argv.pop(0)
    await sdapi.session()
    if len(sys.argv) == 0:
        log.error({ 'interrogate': 'no files specified' })
    for arg in sys.argv:
        if os.path.exists(arg):
            if os.path.isfile(arg):
                await interrogate(arg)
            elif os.path.isdir(arg):
                for root, _dirs, files in os.walk(arg):
                    for f in files:
                        _caption, _keywords, _style = await interrogate(os.path.join(root, f))
            else:
                log.error({ 'interrogate unknown file type': arg })
        else:
            log.error({ 'interrogate file missing': arg })
    await sdapi.close()
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
