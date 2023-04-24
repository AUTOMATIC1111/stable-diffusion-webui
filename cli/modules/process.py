#!/bin/env python
"""
process people images
- check image resolution
- runs detection of face and body
- extracts crop and performs checks:
    - visible: is face or body detected
    - in frame: for face based on box, for body based on number of visible keypoints
    - resolution: is cropped image still of sufficient resolution
    - optionaly upsample and restore face quality
    - blur: is image sharp enough
    - dynamic range: is image bright enough
    - similarity: compares image to all previously processed images to see if its unique enough
- images are resized and optionally squared
- face additionally runs through semantic segmentation to remove background
- if image passes checks  
  image padded and saved as extracted image
- body requires that face is detected and in-frame,  
  but does not have to pass all other checks as body performs its own checks
- runs clip interrogation on extracted images to generate filewords
"""

import os
import sys
import io
import math
import base64
import pathlib
import argparse
import logging
import filetype
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from scipy.stats import beta
sys.path.append(os.path.join(os.path.dirname(__file__)))

from util import log, Map
from sdapi import postsync


params = Map({
    # general settings, do not modify
    'src': '', # source folder
    'dst': '', # destination folder
    'clear_dst': True, # remove all files from destination at the start
    'format': '.jpg', # image format
    'target_size': 512, # target resolution
    'square_images': True, # should output images be squared
    'segmentation_model': 0, # segmentation model 0/general 1/landscape
    'segmentation_background': (192, 192, 192), # segmentation background color
    'blur_samplesize': 60, # sample size to use for blur detection
    'similarity_size': 64, # base similarity detection on reduced images
    # original image processing settings
    'keep_original': False, # keep original image
    # face processing settings
    'extract_face': False, # extract face from image
    'face_score': 0.7, # min face detection score
    'face_pad': 0.1, # pad face image percentage
    'face_model': 1, # which face model to use 0/close-up 1/standard
    'face_blur': False, # check for body blur
    'face_blur_score': 1.5, # max score for face blur detection
    'face_range': False, # check for body blur
    'face_range_score': 0.15, # min score for face dynamic range detection
    'face_restore': False, # attempt to restore face quality
    'face_upscale': False, # attempt to scale small faces
    'face_segmentation': False, # segmentation enabled
    # body processing settings
    'extract_body': False, # extract body from image
    'body_score': 0.9, # min body detection score
    'body_visibility': 0.5, # min visibility score for each detected body part
    'body_parts': 15, # min number of detected body parts with sufficient visibility
    'body_pad': 0.2,  # pad body image percentage
    'body_model': 2, # body model to use 0/low 1/medium 2/high
    'body_blur': False, # check for body blur
    'body_blur_score': 1.8, # max score for body blur detection
    'body_range': False, # check for body blur
    'body_range_score': 0.15, # min score for body dynamic range detection
    'body_segmentation': False, # segmentation enabled
    # similarity detection settings
    'similarity_score': 0.8, # maximum similarity score before image is discarded
    # interrogate settings
    'interrogate_model': ['clip', 'deepdanbooru'], # interrogate models
    'interrogate_captions': True, # write captions to file
    'tag_limit': 5, # number of tags to extract
})
face_model = None
body_model = None
segmentation_model = None


def detect_blur(image):
    # based on <https://github.com/karthik9319/Blur-Detection/>
    bw = ImageOps.grayscale(image)
    cx, cy = image.size[0] // 2, image.size[1] // 2
    fft = np.fft.fft2(bw)
    fftShift = np.fft.fftshift(fft)
    fftShift[cy - params.blur_samplesize: cy + params.blur_samplesize, cx - params.blur_samplesize: cx + params.blur_samplesize] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    mean = round(np.mean(magnitude), 2)
    return mean


def detect_dynamicrange(image):
    # based on <https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10>
    data = np.asarray(image)
    image = np.float32(data)
    RGB = [0.299, 0.587, 0.114]
    height, width = image.shape[:2]
    brightness_image = np.sqrt(image[..., 0] ** 2 * RGB[0] + image[..., 1] ** 2 * RGB[1] + image[..., 2] ** 2 * RGB[2])
    hist, _ = np.histogram(brightness_image, bins=256, range=(0, 255))
    img_brightness_pmf = hist / (height * width)
    dist = beta(2, 2)
    ys = dist.pdf(np.linspace(0, 1, 256))
    ref_pmf = ys / np.sum(ys)
    dot_product = np.dot(ref_pmf, img_brightness_pmf)
    squared_dist_a = np.sum(ref_pmf ** 2)
    squared_dist_b = np.sum(img_brightness_pmf ** 2)
    res = dot_product / math.sqrt(squared_dist_a * squared_dist_b)
    return round(res, 2)


images = []
def detect_simmilar(image):
    img = image.resize((params.similarity_size, params.similarity_size))
    img = ImageOps.grayscale(img)
    data = np.array(img)
    similarity = 0
    for i in images:
        val = ssim(data, i, data_range=255, channel_axis=None, gradient=False, full=False)
        if val > similarity:
            similarity = val
    images.append(data)
    return similarity


def segmentation(image):
    global segmentation_model
    if segmentation_model is None:
        segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=params.segmentation_model)
    data = np.array(image)
    results = segmentation_model.process(data)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    background = np.zeros(data.shape, dtype=np.uint8)
    background[:] = params.segmentation_background
    data = np.where(condition, data, background) # consider using a joint bilateral filter instead of pure combine
    segmented = Image.fromarray(data)
    return segmented


def extract_face(img):
    if not params.extract_face:
        return None, True
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    scale = max(img.size[0], img.size[1]) / params.target_size
    resized = img.copy()
    resized.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    global face_model
    if face_model is None:
        face_model = mp.solutions.face_detection.FaceDetection(min_detection_confidence=params.face_score, model_selection=params.face_model)
    results = face_model.process(np.array(resized))
    if results.detections is None:
        return None, False
    box = results.detections[0].location_data.relative_bounding_box
    if box.xmin < 0 or box.ymin < 0 or (box.width - box.xmin) > 1 or (box.height - box.ymin) > 1:
        log.info({ 'process face skip': 'out of frame' })
        return None, False
    x = (box.xmin - params.face_pad / 2) * resized.width
    y = (box.ymin - params.face_pad / 2)* resized.height
    w = (box.width + params.face_pad) * resized.width
    h = (box.height + params.face_pad) * resized.height
    cx = x + w / 2
    cy = y + h / 2
    l = max(w, h) / 2
    square = [scale * (cx - l), scale * (cy - l), scale * (cx + l), scale * (cy + l)]
    square = [max(square[0], 0), max(square[1], 0), min(square[2], img.width), min(square[3], img.height)]
    cropped = img.crop(tuple(square))

    upscale = 1
    if params.face_restore or params.face_upscale:
        if (cropped.size[0] < params.target_size or cropped.size[1] < params.target_size) and params.face_upscale:
            upscale = 2
        kwargs = Map({
            'image': encode(cropped),
            'upscaler_1': 'SwinIR_4x' if params.face_upscale else None,
            'codeformer_visibility': 1.0 if params.face_restore else 0.0,
            'codeformer_weight': 0.15 if params.face_restore else 0.0,
            'upscaling_resize': upscale,
        })
        original = [cropped.size[0], cropped.size[1]]
        res = postsync('/sdapi/v1/extra-single-image', kwargs)
        if 'image' not in res:
            log.error({ 'process face': 'upscale failed' })
            raise ValueError('upscale failed')
        cropped = Image.open(io.BytesIO(base64.b64decode(res['image'])))
        kwargs.image = [cropped.size[0], cropped.size[1]]
        upscaled = [cropped.size[0], cropped.size[1]]
        upscale = False if upscale == 1 else { 'original': original, 'upscaled': upscaled }
        log.info({ 'process face restore': params.face_restore, 'upscale': upscale })

    if cropped.size[0] < params.target_size and cropped.size[1] < params.target_size:
        log.info({ 'process face skip': 'low resolution', 'size': [cropped.size[0], cropped.size[1]] })
        return None, True
    cropped.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    if params.square_images:
        squared = Image.new('RGB', (params.target_size, params.target_size))
        squared.paste(cropped, ((params.target_size - cropped.width) // 2, (params.target_size - cropped.height) // 2))
        if params.face_segmentation:
           squared = segmentation(squared)
    else:
        squared = cropped

    if params.face_blur:
        blur = detect_blur(squared)
        if blur > params.face_blur_score:
            log.info({ 'process face skip': 'blur check fail', 'blur': blur })
            return None, True
        else:
            log.debug({ 'process face blur': blur })

    if params.face_range:
        range = detect_dynamicrange(squared)
        if range < params.face_range_score:
            log.info({ 'process face skip': 'dynamic range check fail', 'range': range })
            return None, True
        else:
            log.debug({ 'process face dynamic range': range })

    similarity = detect_simmilar(squared)
    if similarity > params.similarity_score:
        log.info({ 'process face skip': 'similarity check fail', 'score': round(similarity, 2) })
        return None, True

    return squared, True


def extract_body(img):
    if not params.extract_body:
        return None, True
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    scale = max(img.size[0], img.size[1]) / params.target_size
    resized = img.copy()
    resized.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    global body_model
    if body_model is None:
        body_model = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=params.body_score, model_complexity=params.body_model)
    results = body_model.process(np.array(resized))
    if results.pose_landmarks is None:
        return None, False
    x = [resized.width * (i.x - params.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > params.body_visibility]
    y = [resized.height * (i.y - params.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > params.body_visibility]
    if len(x) < params.body_parts:
        log.info({ 'process body skip': 'insufficient body parts', 'detected': len(x) })
        return None, True
    w = max(x) - min(x) + resized.width * params.body_pad
    h = max(y) - min(y) + resized.height * params.body_pad
    cx = min(x) + w / 2
    cy = min(y) + h / 2
    l = max(w, h) / 2
    square = [scale * (cx - l), scale * (cy - l), scale * (cx + l), scale * (cy + l)]
    square = [max(square[0], 0), max(square[1], 0), min(square[2], img.width), min(square[3], img.height)]
    cropped = img.crop(tuple(square))
    if cropped.size[0] < params.target_size and cropped.size[1] < params.target_size:
        log.info({ 'process body skip': 'low resolution', 'size': [cropped.size[0], cropped.size[1]] })
        return None, True
    cropped.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    if params.square_images:
        squared = Image.new('RGB', (params.target_size, params.target_size))
        squared.paste(cropped, ((params.target_size - cropped.width) // 2, (params.target_size - cropped.height) // 2))
        if params.body_segmentation:
           squared = segmentation(squared)
    else:
        squared = cropped

    if params.body_blur:
        blur = detect_blur(squared)
        if blur > params.body_blur_score:
            log.info({ 'process body skip': 'blur check fail', 'blur': blur })
            return None, True
        else:
            log.debug({ 'process body blur': blur })

    if params.body_range:
        range = detect_dynamicrange(squared)
        if range < params.body_range_score:
            log.info({ 'process body skip': 'dynamic range check fail', 'range': range })
            return None, True
        else:
            log.debug({ 'process body dynamic range': range })

    similarity = detect_simmilar(squared)
    if similarity > params.similarity_score:
        log.info({ 'process body skip': 'similarity check fail', 'score': round(similarity, 2) })
        return None, True

    return squared, True


def save_original(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    resized = img.copy()
    resized.thumbnail((params.target_size, params.target_size), Image.HAMMING)
    if params.square_images:
        squared = Image.new('RGB', (params.target_size, params.target_size))      
        squared.paste(resized, ((params.target_size - resized.width) // 2, (params.target_size - resized.height) // 2))
    else:
        squared = resized
    return squared


def encode(img):
    with io.BytesIO() as stream:
        img.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def interrogate(img, fn, intag = None):
    if len(params.interrogate_model) == 0:
        return
    caption = ''
    tags = []
    for model in params.interrogate_model:
        json = Map({ 'image': encode(img), 'model': model })
        res = postsync('/sdapi/v1/interrogate', json)
        if model == 'clip':
            caption = res.caption if 'caption' in res else ''
            caption = caption.split(',')[0].replace('a ', '')
            if intag is not None:
                caption = intag + ', ' + caption
        if model == 'deepdanbooru':
            tag = res.caption if 'caption' in res else ''
            tags = tag.split(',')
            tags = [t.replace('(', '').replace(')', '').replace('\\', '').split(':')[0].strip() for t in tags]
            if intag is not None:
                for t in intag.split(',')[::-1]:
                    tags.insert(0, t.strip())
        if params.interrogate_captions:
            file = fn.replace(params.format, '.txt')
            f = open(file, 'w')
            f.write(caption)
            f.close()
    pos = 0 if len(tags) == 0 else 1
    tags.insert(pos, caption.split(' ')[1])
    if len(tags) > params.tag_limit:
        tags = tags[:params.tag_limit]
    log.info({ 'interrogate': caption, 'tags': tags })
    return caption, tags


i = {}
metadata = Map({})

# entry point when used as module
def process_file(f: str, dst: str = None, preview: bool = False, offline: bool = False, txt = None, tag = None, opts = []):
    def save(img, f, what):
        i[what] = i.get(what, 0) + 1
        if dst is None:
            dir = os.path.dirname(f)
        else:
            dir = dst
        base = os.path.basename(f).split('.')[0]
        parent = os.path.basename(pathlib.Path(dir))
        basename = str(i[what]).rjust(3, '0') + '-' + what + '-' + base
        fn = basename + params.format
        # log.debug({ 'save': fn })
        caption = ''
        tags = ''
        if not preview:
            img.save(os.path.join(dir, fn))
            if not offline:
                caption, tags = interrogate(img, os.path.join(dir, fn), tag)
        metadata[os.path.join(parent, basename)] = { 'caption': caption, 'tags': ','.join(tags) }
        return fn

    # overrides
    if len(opts) > 0:
        params.keep_original = True if 'original' in opts else False
        params.extract_face = True if 'face' in opts else False
        params.extract_body = True if 'body' in opts else False
        params.face_blur = True if 'blur' in opts else False
        params.body_blur = True if 'blur' in opts else False
        params.face_range = True if 'range' in opts else False
        params.body_range = True if 'range' in opts else False
        params.face_upscale = True if 'upscale' in opts else False
        params.face_restore = True if 'restore' in opts else False

    log.info({ 'processing': f })
    try:
        image = Image.open(f)
    except Exception as err:
        log.error({ 'image': f, 'error': err })
        return 0, {}

    image = ImageOps.exif_transpose(image) # rotate image according to EXIF orientation
    if txt is not None:
        params.interrogate_captions = txt

    if image.width < 512 or image.height < 512:
        log.info({ 'process skip': 'low resolution', 'resolution': [image.width, image.height] })
        return 0, {}
    log.debug({ 'resolution': [image.width, image.height], 'mp': round((image.width * image.height) / 1024 / 1024, 1) })

    face, ok = extract_face(image)
    if face is not None:
        fn = save(face, f, 'face')
        log.info({ 'extract face': fn })
    else:
        log.debug({ 'no face': f })

    if not ok:
        return 0, {}

    body, ok = extract_body(image)
    if body is not None:
        fn = save(body, f, 'body')
        log.info({ 'extract body': fn })   
    else:
        log.debug({ 'no body': f })

    if params.keep_original:
        resized = save_original(image)
        fn = save(resized, f, 'original')
        log.info({ 'original': fn })

    image.close()
    return i, metadata

def process_images(src: str, dst: str, args = None):
    params.src = src
    params.dst = dst
    if args is not None:
        params.update(args)
    log.info({ 'processing': params })
    if not os.path.isdir(src):
        log.error({ 'process': 'not a folder', 'src': src })
    else:
        if os.path.isdir(dst) and params.clear_dst:
            log.info({ 'clear dst': dst })
            i = [os.path.join(dst, f) for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f)) and filetype.is_image(os.path.join(dst, f))]
            for f in i:
                os.remove(f)
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
        for root, _sub_dirs, files in os.walk(src):
            for f in files:
                i, _metadata = process_file(os.path.join(root, f), dst)
    return i


def unload_models():
    global face_model
    if face_model is not None:
        face_model = None
    global body_model
    if body_model is not None:
        body_model = None
    global segmentation_model
    if segmentation_model is not None:
        segmentation_model = None


if __name__ == '__main__':
    # log.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description = 'dataset processor')
    parser.add_argument('--output', type=str, required=True, help='folder to store images')
    parser.add_argument('--preview', default=False, action='store_true', help = "run processing but do not store results")
    parser.add_argument('--offline', default=False, action='store_true', help = "run only processing steps that do not require running server")
    parser.add_argument('--debug', default=False, action='store_true', help = "enable debug logging")
    parser.add_argument('input', type=str, nargs='*')
    args = parser.parse_args()
    params.dst = args.output
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.info({ 'processing': params })
    if not os.path.exists(params.dst) and not args.preview:
        pathlib.Path(params.dst).mkdir(parents=True, exist_ok=True)
    files = []
    for loc in args.input:
        if os.path.isfile(loc):
            files.append(loc)
        elif os.path.isdir(loc):
            for root, _sub_dirs, dir in os.walk(loc):
                for f in dir:
                    files.append(os.path.join(root, f))
    for f in files:
        process_file(f, params.dst, args.preview, args.offline)
    log.info({ 'processed': i, 'inputs': len(files) })
    # print(json.dumps(metadata, indent=2))
