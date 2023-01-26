#!/bin/env python
"""
process people images
- check image resolution
- runs detection of face and body
- extracts crop and performs checks:
    - visible: is face or body detected
    - in frame: for face based on box, for body based on number of visible keypoints
    - resolution: is cropped image still of sufficient resolution
    - blur: is image sharp enough
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
import filetype
import base64
import pathlib

import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim

from util import log, Map
from sdapi import postsync


params = Map({
    'src': '', # source folder
    'dst': '', # destination folder
    'extract_face': True, # extract face from image
    'extract_body': True, # extract face from image
    'clear_dst': True, # remove all files from destination at the start
    'target_size': 512, # target resolution
    'square_images': True, # should output images be squared
    'blur_samplesize': 60, # sample size to use for blur detection
    'face_score': 0.7, # min face detection score
    'face_pad': 0.07, # pad face image percentage
    'face_model': 1, # which face model to use 0/close-up 1/standard
    'face_blur_score': 1.4, # max score for face blur detection
    'body_score': 0.9, # min body detection score
    'body_visibility': 0.5, # min visibility score for each detected body part
    'body_parts': 15, # min number of detected body parts with sufficient visibility
    'body_pad': 0.2,  # pad body image percentage
    'body_model': 2, # body model to use 0/low 1/medium 2/high
    'body_blur_score': 1.6, # max score for body blur detection
    'segmentation_face': True, # segmentation enabled
    'segmentation_body': False, # segmentation enabled
    'segmentation_model': 0, # segmentation model 0/general 1/landscape
    'segmentation_background': (192, 192, 192), # segmentation background color
    'similarity_score': 0.6, # maximum similarity score before image is discarded
    'similarity_size': 64, # base similarity detection on reduced images
    'interrogate_model': 'clip' # interrogate model
})


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
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=params.segmentation_model) as selfie_segmentation:
        data = np.array(image)
        results = selfie_segmentation.process(data)
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

    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=params.face_score, model_selection=params.face_model) as face:
        results = face.process(np.array(resized))
    if results.detections is None:
        return None, False
    box = results.detections[0].location_data.relative_bounding_box
    if box.xmin < 0 or box.ymin < 0 or (box.width - box.xmin) > 1 or (box.height - box.ymin) > 1:
        log.info({ 'extract face': 'out of frame' })
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
    if cropped.size[0] < params.target_size and cropped.size[1] < params.target_size:
        log.info({ 'extract face': 'low resolution', 'size': [cropped.size[0], cropped.size[1]] })
        return None, True
    cropped.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    if params.square_images:
        squared = Image.new('RGB', (params.target_size, params.target_size))
        squared.paste(cropped, (0, 0))
        if params.segmentation_face:
           squared = segmentation(squared)
    else:
        squared = cropped

    blur = detect_blur(squared)
    if blur > params.face_blur_score:
        log.info({ 'extract face': 'blur check fail', 'blur': blur })
        return None, True
    else:
        log.debug({ 'extract face blur': blur })

    similarity = detect_simmilar(squared)
    if similarity > params.similarity_score:
        log.info({ 'extract face': 'similarity check fail', 'score': round(similarity, 2) })
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
    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=params.body_score, model_complexity=params.body_model) as pose:
        results = pose.process(np.array(resized))
    if results.pose_landmarks is None:
        return None, False
    x = [resized.width * (i.x - params.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > params.body_visibility]
    y = [resized.height * (i.y - params.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > params.body_visibility]
    if len(x) < params.body_parts:
        log.info({ 'extract body': 'insufficient body parts', 'detected': len(x) })
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
        log.info({ 'extract body': 'low resolution', 'size': [cropped.size[0], cropped.size[1]] })
        return None, True
    cropped.thumbnail((params.target_size, params.target_size), Image.HAMMING)

    if params.square_images:
        squared = Image.new('RGB', (params.target_size, params.target_size))
        squared.paste(cropped, (0, 0))
        if params.segmentation_body:
           squared = segmentation(squared)
    else:
        squared = cropped

    blur = detect_blur(squared)
    if blur > params.body_blur_score:
        log.info({ 'extract body': 'blur check fail', 'blur': blur })
        return None, True
    else:
        log.debug({ 'extract body blur': blur })

    similarity = detect_simmilar(squared)
    if similarity > params.similarity_score:
        log.info({ 'extract body': 'similarity check fail', 'score': similarity })
        return None, True

    return squared, True


def interrogate(img, fn):
    def encode(f):
        with io.BytesIO() as stream:
            img.save(stream, 'JPEG')
            values = stream.getvalue()
            encoded = base64.b64encode(values).decode()
            return encoded

    if params.interrogate_model is None or params.interrogate_model == '':
        return
    json = Map({ 'image': encode(img), 'model': params.interrogate_model })
    res = postsync('/sdapi/v1/interrogate', json)
    caption = res.caption if 'caption' in res else ''
    log.info({ 'interrogate': caption })
    file = fn.replace('.jpg', '.txt')
    f = open(file, 'w')
    f.write(caption)
    f.close()


i = {}
def process_file(f: str, dst: str = None):
    def save(img, f, what):
        i[what] = i.get(what, 0) + 1
        if dst is None:
            dir = os.path.dirname(f)
        else:
            dir = dst
        base = os.path.basename(f).split('.')[0]
        fn = os.path.join(dir, str(i[what]).rjust(3, '0') + '-' + what + '-' + base + '.jpg')
        # log.debug({ 'save': fn })
        img.save(fn)
        interrogate(img, fn)
        return fn

    log.info({ 'processing': f })
    try:
        image = Image.open(f)
    except Exception as err:
        log.error({ 'image': f, 'error': err })
        return

    image = ImageOps.exif_transpose(image) # rotate image according to EXIF orientation

    if image.width < 512 or image.height < 512:
        log.info({ 'skip low resolution': [image.width, image.height], 'file': f })
        return
    log.debug({ 'resolution': [image.width, image.height], 'mp': round((image.width * image.height) / 1024 / 1024, 1) })

    face, ok = extract_face(image)
    if face is not None:
        fn = save(face, f, 'face')
        log.info({ 'extract face': fn })
    else:
        log.debug({ 'no face': f })

    if not ok:
        return

    body, ok = extract_body(image)
    if body is not None:
        fn = save(body, f, 'body')
        log.info({ 'extract body': fn })   
    else:
        log.debug({ 'no body': f })

    image.close()

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
                process_file(os.path.join(root, f), dst)


if __name__ == '__main__':
    # log.setLevel(logging.DEBUG)
    sys.argv.pop(0)
    dst = sys.argv.pop(0)
    params.dst = dst
    log.info({ 'processing': params })
    pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    for loc in sys.argv:
        if os.path.isfile(loc):
            process_file(loc, dst)
        elif os.path.isdir(loc):
            for root, _sub_dirs, files in os.walk(loc):
                for f in files:
                    process_file(os.path.join(root, f), dst)
