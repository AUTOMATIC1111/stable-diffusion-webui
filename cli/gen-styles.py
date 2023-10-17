#!/bin/env python

import io
import json
import base64
import argparse
import requests
from PIL import Image


options = {
    "negative_prompt": "",
    "steps": 20,
    "batch_size": 1,
    "n_iter": 1,
    "seed": -1,
    "sampler_name": "UniPC",
    "cfg_scale": 6,
    "width": 512,
    "height": 512,
    "save_images": False,
    "send_images": True,
}
styles = []


def pil_to_b64(img: Image, size: int, quality: int):
    img = img.convert('RGB')
    img = img.resize((size, size))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    b64encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f'data:image/jpeg;base64,{b64encoded}'


def post(endpoint: str, dct: dict = None):
    req = requests.post(endpoint, json = dct, timeout=300, verify=False)
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'gen-styles.py')
    parser.add_argument('--input', type=str, required=True, help="input text file with one line per prompt")
    parser.add_argument('--output', type=str, required=True, help="output json file")
    parser.add_argument('--nopreviews', default=False, action='store_true', help = 'generate previews')
    parser.add_argument('--prompt', type=str, required=False, default='girl walking in a city', help="applied prompt when generating previews")
    parser.add_argument('--size', type=int, default=128, help="image size for previews")
    parser.add_argument('--quality', type=int, default=35, help="image quality for previews")
    parser.add_argument('--url', type=str, required=False, default='http://127.0.0.1:7860', help="sd.next server url")
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().replace('\n', '')
        if len(line) == 0:
            continue
        print(f'processing: {line}')
        if not args.nopreviews:
            options['prompt'] = f'{line} {args.prompt}'
            data = post(f'{args.url}/sdapi/v1/txt2img', options)
            if 'error' in data:
                print(f'error: {data}')
                continue
            b64str = data['images'][0].split(',',1)[0]
            image = Image.open(io.BytesIO(base64.b64decode(b64str)))
        else:
            image = None
        styles.append({
            'name': line,
            'prompt': line + ' {prompt}',
            'negative': '',
            'extra': '',
            'preview': pil_to_b64(image, args.size, args.quality) if image is not None else '',
        })
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(styles, outfile, indent=2)
