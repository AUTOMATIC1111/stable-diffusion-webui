 # pylint: disable=global-statement
import os
import io
import math
import base64
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from scipy.stats import beta

import util
import sdapi
import options

face_model = None
body_model = None
segmentation_model = None
all_images = []
all_images_by_type = {}


class Result(object):
    def __init__(self, typ: str, fn: str, tag: str = None, requested: list = []):
        self.type = typ
        self.input = fn
        self.output = ''
        self.basename = ''
        self.message = ''
        self.image = None
        self.caption = ''
        self.tag = tag
        self.tags = []
        self.ops = []
        self.steps = requested


def detect_blur(image: Image):
    # based on <https://github.com/karthik9319/Blur-Detection/>
    bw = ImageOps.grayscale(image)
    cx, cy = image.size[0] // 2, image.size[1] // 2
    fft = np.fft.fft2(bw)
    fftShift = np.fft.fftshift(fft)
    fftShift[cy - options.process.blur_samplesize: cy + options.process.blur_samplesize, cx - options.process.blur_samplesize: cx + options.process.blur_samplesize] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    mean = round(np.mean(magnitude), 2)
    return mean


def detect_dynamicrange(image: Image):
    # based on <https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10>
    data = np.asarray(image)
    image = np.float32(data)
    RGB = [0.299, 0.587, 0.114]
    height, width = image.shape[:2] # pylint: disable=unsubscriptable-object
    brightness_image = np.sqrt(image[..., 0] ** 2 * RGB[0] + image[..., 1] ** 2 * RGB[1] + image[..., 2] ** 2 * RGB[2]) # pylint: disable=unsubscriptable-object
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


def detect_simmilar(image: Image):
    img = image.resize((options.process.similarity_size, options.process.similarity_size))
    img = ImageOps.grayscale(img)
    data = np.array(img)
    similarity = 0
    for i in all_images:
        val = ssim(data, i, data_range=255, channel_axis=None, gradient=False, full=False)
        if val > similarity:
            similarity = val
    all_images.append(data)
    return similarity


def segmentation(res: Result):
    global segmentation_model
    if segmentation_model is None:
        segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=options.process.segmentation_model)
    data = np.array(res.image)
    results = segmentation_model.process(data)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    background = np.zeros(data.shape, dtype=np.uint8)
    background[:] = options.process.segmentation_background
    data = np.where(condition, data, background) # consider using a joint bilateral filter instead of pure combine
    segmented = Image.fromarray(data)
    res.image = segmented
    res.ops.append('segmentation')
    return res


def unload():
    global face_model
    if face_model is not None:
        face_model = None
    global body_model
    if body_model is not None:
        body_model = None
    global segmentation_model
    if segmentation_model is not None:
        segmentation_model = None


def encode(img):
    with io.BytesIO() as stream:
        img.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def reset():
    unload()
    global all_images_by_type
    all_images_by_type = {}
    global all_images
    all_images = []


def upscale_restore_image(res: Result, upscale: bool = False, restore: bool = False):
    kwargs = util.Map({
        'image': encode(res.image),
        'codeformer_visibility': 0.0,
        'codeformer_weight': 0.0,
    })
    if res.image.width >= options.process.target_size and res.image.height >= options.process.target_size:
        upscale = False
    if upscale:
        kwargs.upscaler_1 = 'SwinIR_4x'
        kwargs.upscaling_resize = 2
        res.ops.append('upscale')
    if restore:
        kwargs.codeformer_visibility = 1.0
        kwargs.codeformer_weight: 0.2
        res.ops.append('restore')
    if upscale or restore:
        result = sdapi.postsync('/sdapi/v1/extra-single-image', kwargs)
        if 'image' not in result:
            res.message = 'failed to upscale/restore image'
        else:
            res.image = Image.open(io.BytesIO(base64.b64decode(result['image'])))
    return res


def interrogate_image(res: Result, tag: str = None):
    caption = ''
    tags = []
    for model in options.process.interrogate_model:
        json = util.Map({ 'image': encode(res.image), 'model': model })
        result = sdapi.postsync('/sdapi/v1/interrogate', json)
        if model == 'clip':
            caption = result.caption if 'caption' in result else ''
            caption = caption.split(',')[0].replace('a ', '')
            if tag is not None:
                caption = res.tag + ', ' + caption
        if model == 'deepdanbooru':
            tag = result.caption if 'caption' in result else ''
            tags = tag.split(',')
            tags = [t.replace('(', '').replace(')', '').replace('\\', '').split(':')[0].strip() for t in tags]
            if tag is not None:
                for t in res.tag.split(',')[::-1]:
                    tags.insert(0, t.strip())
    pos = 0 if len(tags) == 0 else 1
    tags.insert(pos, caption.split(' ')[1])
    if len(tags) > options.process.tag_limit:
        tags = tags[:options.process.tag_limit]
    res.caption = caption
    res.tags = tags
    res.ops.append('interrogate')
    return res


def resize_image(res: Result):
    resized = res.image
    resized.thumbnail((options.process.target_size, options.process.target_size), Image.HAMMING)
    res.image = resized
    res.ops.append('resize')
    return res


def square_image(res: Result):
    size = max(res.image.width, res.image.height)
    squared = Image.new('RGB', (size, size))
    squared.paste(res.image, ((size - res.image.width) // 2, (size - res.image.height) // 2))
    res.image = squared
    res.ops.append('square')
    return res


def process_face(res: Result):
    res.ops.append('face')
    global face_model
    if face_model is None:
        face_model = mp.solutions.face_detection.FaceDetection(min_detection_confidence=options.process.face_score, model_selection=options.process.face_model)
    results = face_model.process(np.array(res.image))
    if results.detections is None:
        res.message = 'no face detected'
        res.image = None
        return res
    box = results.detections[0].location_data.relative_bounding_box
    if box.xmin < 0 or box.ymin < 0 or (box.width - box.xmin) > 1 or (box.height - box.ymin) > 1:
        res.message = 'face out of frame'
        res.image = None
        return res
    x = max(0, (box.xmin - options.process.face_pad / 2) * res.image.width)
    y = max(0, (box.ymin - options.process.face_pad / 2)* res.image.height)
    w = min(res.image.width, (box.width + options.process.face_pad) * res.image.width)
    h = min(res.image.height, (box.height + options.process.face_pad) * res.image.height)
    x = max(0, x)
    res.image = res.image.crop((x, y, x + w, y + h))
    return res


def process_body(res: Result):
    res.ops.append('body')
    global body_model
    if body_model is None:
        body_model = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=options.process.body_score, model_complexity=options.process.body_model)
    results = body_model.process(np.array(res.image))
    if results.pose_landmarks is None:
        res.message = 'no body detected'
        res.image = None
        return res
    x0 = [res.image.width * (i.x - options.process.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > options.process.body_visibility]
    y0 = [res.image.height * (i.y - options.process.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > options.process.body_visibility]
    x1 = [res.image.width * (i.x + options.process.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > options.process.body_visibility]
    y1 = [res.image.height * (i.y + options.process.body_pad / 2) for i in results.pose_landmarks.landmark if i.visibility > options.process.body_visibility]
    if len(x0) < options.process.body_parts:
        res.message = f'insufficient body parts detected: {len(x0)}'
        res.image = None
        return res
    res.image = res.image.crop((max(0, min(x0)), max(0, min(y0)), min(res.image.width, max(x1)), min(res.image.height, max(y1))))
    return res


def process_original(res: Result):
    res.ops.append('original')
    return res


def save_image(res: Result, folder: str):
    if res.image is None or folder is None:
        return res
    all_images_by_type[res.type] = all_images_by_type.get(res.type, 0) + 1
    res.basename = os.path.basename(res.input).split('.')[0]
    res.basename = str(all_images_by_type[res.type]).rjust(3, '0') + '-' + res.type + '-' + res.basename
    res.basename = os.path.join(folder, res.basename)
    res.output = res.basename + options.process.format
    res.image.save(res.output)
    res.image.close()
    res.ops.append('save')
    return res


def file(filename: str, folder: str, tag = None, requested = []):
    # initialize result dict
    res = Result(fn = filename, typ='unknown', tag=tag, requested = requested)
    # open image
    try:
        res.image = Image.open(filename)
        if res.image.mode == 'RGBA':
            res.image = res.image.convert('RGB')
        res.image = ImageOps.exif_transpose(res.image) # rotate image according to EXIF orientation
    except Exception as e:
        res.message = f'error opening: {e}'
        return res
    # primary steps
    if 'face' in requested:
        res.type = 'face'
        res = process_face(res)
    elif 'body' in requested:
        res.type = 'body'
        res = process_body(res)
    elif 'original' in requested:
        res.type = 'original'
        res = process_original(res)
    # validation steps
    if res.image is None:
        return res
    if 'blur' in requested:
        res.ops.append('blur')
        val = detect_blur(res.image)
        if val > options.process.blur_score:
            res.message = f'blur check failed: {val}'
            res.image = None
    if 'range' in requested:
        res.ops.append('range')
        val = detect_dynamicrange(res.image)
        if val < options.process.range_score:
            res.message = f'dynamic range check failed: {val}'
            res.image = None
    if 'similarity' in requested:
        res.ops.append('similarity')
        val = detect_simmilar(res.image)
        if val > options.process.similarity_score:
            res.message = f'dynamic range check failed: {val}'
            res.image = None
    if res.image is None:
        return res
    # post processing steps
    res = upscale_restore_image(res, 'upscale' in requested, 'restore' in requested)
    if res.image.width < options.process.target_size or res.image.height < options.process.target_size:
        res.message = f'low resolution: [{res.image.width}, {res.image.height}]'
        res.image = None
        return res
    if 'interrogate' in requested:
        res = interrogate_image(res, tag)
    if 'resize' in requested:
        res = resize_image(res)
    if 'square' in requested:
        res = square_image(res)
    if 'segment' in requested:
        res = segmentation(res)
    # finally save image
    res = save_image(res, folder)
    return res
