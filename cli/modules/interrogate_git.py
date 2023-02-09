#!/bin/env python

import os
import json
import argparse
import torch
import filetype
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from util import log, Map


git_processor = None
git_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


options = Map({
    'input': '',
    'model': 'microsoft/git-large-textcaps',
    'length': 256,
    'json': '',
    'txt': False,
    'tag': '',
})

def cleanup(s: str):
    s = s.split('"')[0].split('.')[0].split(' that')[0]
    s = s.split(' with a letter')[0].split(' with the number')[0].split(' with the word')[0]
    return s.replace('a ', '')


def load_model(args):
    global git_processor
    global git_model
    if git_processor is None:
        git_processor = AutoProcessor.from_pretrained(args.model)
    if git_model is None:
        git_model = AutoModelForCausalLM.from_pretrained(args.model)
    git_model.to(device)
    log.info( { 'interrogate loaded model': args.model })


def interrogate_files(params, files):
    args = Map({**options, **params})
    data = [f for f in files if filetype.is_image(f)]
    log.info({ 'interrogate files': len(files), 'images': len(data), 'args': args })
    load_model(args)
    metadata = {}
    for image_path in data:
        image = Image.open(image_path)
        inputs = git_processor(images=[image], return_tensors="pt").to(device)
        generated_ids = git_model.generate(pixel_values=inputs.pixel_values, max_length=args.length)
        caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption = cleanup(caption)
        tags = ''
        if args.tag != '':
            tags += args.tag + ','
        tags += caption.split(' ')[0]
        if args.txt:
            with open(os.path.splitext(image_path)[0] + '.txt', "wt", encoding='utf-8') as f:
                f.write(caption + "\n")
        metadata[image_path] = { 'caption': caption, 'tags': tags }
        log.info({ 'interrogate image': image_path, 'caption': caption, 'tags': tags })

    git_model.to('cpu')
    if args.json != '':
        with open(args.json, "wt", encoding='utf-8') as f:
            f.write(json.dumps(metadata, indent=2) + "\n")
    return metadata


def unload_git():
    global git_processor
    global git_model
    del git_processor
    del git_model
    git_processor = None
    git_model = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'image interrogate')
    parser.add_argument('input', type=str, nargs='*', help='input file or directory')
    parser.add_argument("--model", type=str, default="microsoft/git-large-textcaps", help="model id for GIT in HuggingFace")
    parser.add_argument("--length", type=int, default=256, help="max length of caption")
    parser.add_argument("--json", type=str, default='', help="output json file")
    parser.add_argument("--tag", type=str, default='', help="append tag")
    parser.add_argument('--txt', default = False, action='store_true', help = "write captions to text files")
    params = parser.parse_args()
    log.info({ 'interrogate args': vars(params) })
    if len(params.input) == 0:
        parser.print_help()
        exit(1)
    files = []
    for loc in params.input:
        if os.path.isfile(loc):
            files.append(loc)
        elif os.path.isdir(loc):
            for root, _sub_dirs, dir in os.walk(loc):
                files = [os.path.join(root, f) for f in dir]
    metadata = interrogate_files(vars(params), files)
