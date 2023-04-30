#!/bin/env python

import os
import gc
import json
import time
import argparse
import torch
import filetype
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from util import log, Map


model = None
processor = None
extractor = None
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

options = Map({
    'input': '',
    'min': 8,
    'max': 256,
    'beams': 1,
    'json': '',
    'txt': False,
    'tag': '',
    'git': True,
    'blip': True,
    'precision': 'fp16',
    'model': 'git',
})


def cleanup(s: str):
    s = s.split('"')[0].split('.')[0].split(' that')[0]
    s = s.split(' with a letter')[0].split(' with the number')[0].split(' with the word')[0]
    s = s.replace('arafed image of ', '')
    return s.replace('a ', '')


def load_model(args):
    global model
    global processor
    global extractor
    transformers.logging.set_verbosity_error()
    if args.model == 'git':
        model_name = "microsoft/git-large-textcaps"
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_name, torch_dtype=dtype)
            log.info( { 'interrogate loaded model': model_name })
    elif args.model == 'blip':
        model_name = "Salesforce/blip-image-captioning-large"
        if model is None:
            model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            processor = BlipProcessor.from_pretrained(model_name, torch_dtype=dtype)
            log.info( { 'interrogate loaded model': model_name })
    elif args.model == 'vit':      
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        if model is None:
            model = VisionEncoderDecoderModel.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            extractor = ViTFeatureExtractor.from_pretrained(model_name, torch_dtype=dtype)
            processor = AutoTokenizer.from_pretrained(model_name, torch_dtype=dtype)
            log.info( { 'interrogate loaded model': model_name })
    else:
        log.info( { 'interrogate unknown model': args.model })


def interrogate_files(params, files):
    args = Map({**options, **params})
    data = [f for f in files if filetype.is_image(f)]
    log.info({ 'interrogate files': len(files), 'images': len(data), 'args': args })
    load_model(args)
    metadata = {}
    for image_path in data:
        image = Image.open(image_path).convert('RGB')
        caption = ''
        if args.model == 'git':
            inputs = processor(images=[image], return_tensors="pt").to(device)
            ids = model.generate(pixel_values=inputs.pixel_values, num_beams=args.beams, min_length=args.min, max_length=args.max)
            caption = processor.batch_decode(ids, skip_special_tokens=True)[0]
        elif args.model == 'blip':
            inputs = processor(image, return_tensors="pt").to(device, dtype)
            ids = model.generate(**inputs, num_beams=args.beams, min_length=args.min, max_length=args.max)
            caption = processor.decode(ids[0], skip_special_tokens=True)
        elif args.model == 'vit':
            inputs = extractor(images=[image], return_tensors="pt").pixel_values.to(device)
            ids = model.generate(inputs, num_beams=args.beams, min_length=args.min, max_length=args.max)
            caption = processor.batch_decode(ids, skip_special_tokens=True)[0]
        else:
            log.error({ 'interrogate unknown model': args.model })

        caption = cleanup(caption)
        tags = ''
        if args.tag != '':
            tags += args.tag + ','
        tags += caption.split(' ')[0]
        if args.txt:
            with open(os.path.splitext(image_path)[0] + '.txt', "wt", encoding='utf-8') as f:
                f.write(caption + "\n")
        metadata[image_path] = { 'caption': caption, 'tags': tags }
        log.info({ 'interrogate image': image_path, 'moodel': args.model, 'caption': caption, 'tags': tags })

    if args.json != '':
        with open(args.json, "wt", encoding='utf-8') as f:
            f.write(json.dumps(metadata, indent=2) + "\n")
    return metadata


def unload_model():
    global processor
    global model
    global extractor
    if model is not None:
        del model
        model = None
    if processor is not None:
        del processor
        processor = None
    if extractor is not None:
        del extractor
        extractor = None
    gc.collect()
    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'image interrogate')
    parser.add_argument('input', type=str, nargs='*', help='input file or directory')
    parser.add_argument('--model', default = 'git', choices = ['git', 'blip', 'vit'], help = "which model to use")
    parser.add_argument("--min", type=int, default=8, help="min length of caption")
    parser.add_argument("--max", type=int, default=256, help="max length of caption")
    parser.add_argument("--beams", type=int, default=1, help="number of beams to use")
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
    t0 = time.time()
    metadata = interrogate_files(vars(params), files)
    t1 = time.time()
    log.info({ 'interrogate files': len(files), 'time': round(t1 - t0, 2) })
    unload_model()
