#!/bin/env python

import io
import sys
import json
import base64
import requests
from PIL import Image


url = 'http://127.0.0.1:7860'
size = 128
quality = 35
prompt = 'woman walking in a city'
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


def pil_to_b64(img: Image):
    img = img.convert('RGB')
    img = img.resize((size, size))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    b64encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f'data:image/jpeg;base64,{b64encoded}'


def post(endpoint: str, dct: dict = None):
    req = requests.post(f'{url}{endpoint}', json = dct, timeout=300, verify=False)
    if req.status_code != 200:
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        return req.json()


if __name__ == '__main__':
    sys.argv.pop(0)
    if len(sys.argv) < 2:
        print('gen-styles.py <input text file> <output json file>')
        sys.exit(1)
    with open(sys.argv[0], encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().replace('\n', '')
        print(line)
        options['prompt'] = f'{line} {prompt}'
        data = post('/sdapi/v1/txt2img', options)
        b64str = data['images'][0].split(',',1)[0]
        image = Image.open(io.BytesIO(base64.b64decode(b64str)))
        styles.append({
            'name': line,
            'prompt': line + ' {prompt}',
            'negative': '',
            'extra': '',
            'preview': pil_to_b64(image)
        })
        with open(sys.argv[1], 'w', encoding='utf-8') as outfile:
            json.dump(styles, outfile, indent=2)
