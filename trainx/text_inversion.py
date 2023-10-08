#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/7 10:21 AM
# @Author  : wangdongming
# @Site    : 
# @File    : text_inversion.py
# @Software: Hifive
import os
import re
import numpy as np
import cv2
import math
import requests
import tqdm
from PIL import Image, ImageOps
from enum import Enum
from loguru import logger
from typing import Tuple, List, Dict
from modules import paths, shared, images, deepbooru
from modules.textual_inversion import autocrop


OBS_WD_BASE_URL = 'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/wd-tagger/'.rstrip('/')

tag_escape_pattern = re.compile(r'([\\()])')


# DanBooru IMage Utility functions

def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class WaifuDiffusionInterrogator:
    def __init__(
            self,
            name: str,
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            **kwargs
    ) -> None:
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs
        self.model = None
        self.tags = None
        self.name = name

    def download(self):

        def http_down(url, local):
            print(f'>> download {url} to {local}')
            resp = requests.get(url, timeout=10)
            if resp.ok:
                with open(local, "wb+") as f:
                    f.write(resp.content)

        dir_path = os.path.join(paths.models_path, 'tag_models', self.name)
        revision = self.kwargs.get('revision', '')

        # wd-tagger/wd14-vit-v2/main/model.onnx
        # wd-tagger/wd14-vit-v2/v2.0/model.onnx
        if not revision:
            revision = 'main'
        model_path = os.path.join(dir_path, self.model_path, revision)
        tags_path = os.path.join(dir_path, self.tags_path, revision)
        relative_path = os.path.join(self.name, revision)

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.isfile(model_path):
            url = f"{OBS_WD_BASE_URL}/{relative_path}/{self.model_path}"
            http_down(url, model_path)
        if not os.path.isfile(tags_path):
            url = f"{OBS_WD_BASE_URL}/{relative_path}/{self.tags_path}"
            http_down(url, tags_path)

        return tags_path, model_path

    def load(self) -> None:
        from onnxruntime import InferenceSession
        import pandas as pd

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        tags_path, model_path = self.download()
        logger.info(f'Load wd model from:{model_path}')

        self.model = InferenceSession(str(model_path), providers=providers)
        logger.info(f'> Loaded wd model from:{model_path}')

        self.tags = pd.read_csv(tags_path)

    def interrogate(
            self,
            image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)
        # print(tags)

        return ratings, tags

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            # print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags
        self.model = None
        self.tags = None
        return unloaded

    @staticmethod
    def postprocess_tags(
            tags: Dict[str, float],
            threshold=0.35,
            additional_tags: List[str] = None,
            exclude_tags: List[str] = None,
            sort_by_alphabetical_order=False,
            add_confident_as_weight=False,
            replace_underscore=False,
            replace_underscore_excludes: List[str] = None,
            escape_tag=False
    ) -> Dict[str, float]:

        additional_tags = additional_tags or []
        exclude_tags = exclude_tags or []
        replace_underscore_excludes = replace_underscore_excludes or []
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                    c >= threshold
                    and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags


interrogators = {
    # 'wd14-convnextv2-v2': WaifuDiffusionInterrogator(
    #     'wd14-convnextv2-v2',
    #     repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    #     revision='v2.0'
    # ),
    'wd14-vit-v2': WaifuDiffusionInterrogator(
        'wd14-vit-v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
        revision='v2.0'
    ),
    # 'wd14-convnext-v2': WaifuDiffusionInterrogator(
    #     'wd14-convnext-v2',
    #     repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
    #     revision='v2.0'
    # ),
    # 'wd14-swinv2-v2': WaifuDiffusionInterrogator(
    #     'wd14-swinv2-v2',
    #     repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
    #     revision='v2.0'
    # ),
    # 'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator(
    #     'wd14-convnextv2-v2',
    #     repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    # ),
    # 'wd14-vit-v2-git': WaifuDiffusionInterrogator(
    #     'wd14-vit-v2-git',
    #     repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
    # ),
    # 'wd14-convnext-v2-git': WaifuDiffusionInterrogator(
    #     'wd14-convnext-v2-git',
    #     repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
    # ),
    # 'wd14-swinv2-v2-git': WaifuDiffusionInterrogator(
    #     'wd14-swinv2-v2-git',
    #     repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    # ),
    # 'wd14-vit': WaifuDiffusionInterrogator(
    #     'wd14-vit',
    #     repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
    # 'wd14-convnext': WaifuDiffusionInterrogator(
    #     'wd14-convnext',
    #     repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    # ),
}


def preprocess(id_task, process_src, process_dst, process_width, process_height, preprocess_txt_action,
               process_keep_original_size, process_flip,
               process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5, overlap_ratio=0.2,
               process_focal_crop=False, process_focal_crop_face_weight=0.9, process_focal_crop_entropy_weight=0.3,
               process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False, process_multicrop=None,
               process_multicrop_mindim=None, process_multicrop_maxdim=None, process_multicrop_minarea=None,
               process_multicrop_maxarea=None, process_multicrop_objective=None, process_multicrop_threshold=None):
    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            deepbooru.model.start()

        preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action,
                        process_keep_original_size, process_flip,
                        process_split, process_caption, process_caption_deepbooru, split_threshold, overlap_ratio,
                        process_focal_crop, process_focal_crop_face_weight, process_focal_crop_entropy_weight,
                        process_focal_crop_edges_weight, process_focal_crop_debug, process_multicrop,
                        process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea,
                        process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)

    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.model.stop()


def listfiles(dirname):
    return os.listdir(dirname)


class PreprocessTxtAction(Enum):
    Prepend = 'prepend'
    Append = 'append'
    Copy = 'copy'
    Ignore = 'ignore'


class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    process_caption = False
    process_caption_deepbooru = False
    process_caption_wd_interrogator = None
    process_caption_wd_threshold = 0.35
    preprocess_txt_action = None

    def do_caption(self, existing_caption=None):
        '''
        忽略已存在TAG
        '''
        if not existing_caption:
            return True
        return self.preprocess_txt_action in [
            PreprocessTxtAction.Prepend.value, PreprocessTxtAction.Ignore.value, PreprocessTxtAction.Append.value
        ]


def save_pic_with_caption(image, index, params: PreprocessParams, existing_caption=None):
    caption = ""

    if params.process_caption and params.do_caption(existing_caption):
        caption += shared.interrogator.generate_caption(image)

    if params.process_caption_deepbooru and params.do_caption(existing_caption):
        if len(caption) > 0:
            caption += ", "
        caption += deepbooru.model.tag_multi(image)

    if params.process_caption_wd_interrogator and params.do_caption(existing_caption):
        if len(caption) > 0:
            caption += ", "

            ratings, tags = params.process_caption_wd_interrogator.interrogate(image)
            processed_tags = WaifuDiffusionInterrogator.postprocess_tags(
                tags,
                threshold=params.process_caption_wd_threshold
            )
            caption += ', '.join(processed_tags)

    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.preprocess_txt_action == 'prepend' and existing_caption:
        caption = existing_caption + ',' + caption
    elif params.preprocess_txt_action == 'append' and existing_caption:
        caption = caption + ',' + existing_caption
    elif params.preprocess_txt_action == 'copy' and existing_caption:
        caption = existing_caption

    caption = caption.strip()

    if len(caption) > 0:
        with open(os.path.join(params.dstdir, f"{basename}.txt"), "w", encoding="utf8") as file:
            file.write(caption)

    params.subindex += 1


def save_pic(image, index, params, existing_caption=None):
    save_pic_with_caption(image, index, params, existing_caption=existing_caption)

    if params.flip:
        save_pic_with_caption(ImageOps.mirror(image), index, params, existing_caption=existing_caption)


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted


# not using torchvision.transforms.CenterCrop because it doesn't allow float regions
def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    return wh and center_crop(image, *wh)


def preprocess_work(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_keep_original_size,
                    process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5,
                    overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
                    process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                    process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                    process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                    process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None,
                    caption_wd_interrogator=None, caption_wd_threshold=0.35):
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = listfiles(src)

    shared.state.job = "preprocess"
    shared.state.textinfo = "Preprocessing..."
    shared.state.job_count = len(files)

    params = PreprocessParams()
    params.dstdir = dst
    params.flip = process_flip

    params.preprocess_txt_action = preprocess_txt_action
    params.process_caption_wd_interrogator = caption_wd_interrogator
    params.process_caption_wd_threshold = caption_wd_threshold
    params.process_caption = process_caption
    params.process_caption_deepbooru = process_caption_deepbooru

    pbar = tqdm.tqdm(files)
    for index, imagefile in enumerate(pbar):
        params.subindex = 0
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception:
            continue

        description = f"Preprocessing [Image {index}/{len(files)}]"
        pbar.set_description(description)
        shared.state.textinfo = description

        params.src = filename

        existing_caption = None
        existing_caption_filename = f"{os.path.splitext(filename)[0]}.txt"
        if os.path.exists(existing_caption_filename):
            with open(existing_caption_filename, 'r', encoding="utf8") as file:
                existing_caption = file.read()

        if shared.state.interrupted:
            break

        if img.height > img.width:
            ratio = (img.width * height) / (img.height * width)
            inverse_xy = False
        else:
            ratio = (img.height * width) / (img.width * height)
            inverse_xy = True

        process_default_resize = True

        if process_split and ratio < 1.0 and ratio <= split_threshold:
            for splitted in split_pic(img, inverse_xy, width, height, overlap_ratio):
                save_pic(splitted, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                dnn_model_path = autocrop.download_and_cache_models(os.path.join(paths.models_path, "opencv"))
            except Exception as e:
                print(
                    "Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.",
                    e)

            autocrop_settings = autocrop.Settings(
                crop_width=width,
                crop_height=height,
                face_points_weight=process_focal_crop_face_weight,
                entropy_points_weight=process_focal_crop_entropy_weight,
                corner_points_weight=process_focal_crop_edges_weight,
                annotate_image=process_focal_crop_debug,
                dnn_model_path=dnn_model_path,
            )
            for focal in autocrop.crop_image(img, autocrop_settings):
                save_pic(focal, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_multicrop:
            cropped = multicrop_pic(img, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea,
                                    process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)
            if cropped is not None:
                save_pic(cropped, index, params, existing_caption=existing_caption)
            else:
                print(
                    f"skipped {img.width}x{img.height} image {filename} (can't find suitable size within error threshold)")
            process_default_resize = False

        if process_keep_original_size:
            save_pic(img, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_default_resize:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index, params, existing_caption=existing_caption)

        shared.state.nextjob()
        if callable(progress_cb):
            current_progress = (index + 1) * 100 / len(files)
            if current_progress % 5 == 0:
                progress_cb(current_progress)


def preprocess_sub_dir(process_src,
                       process_dst,
                       process_width,
                       process_height,
                       preprocess_txt_action,
                       process_keep_original_size,
                       process_flip,
                       process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5,
                       overlap_ratio=0.2,
                       process_focal_crop=False, process_focal_crop_face_weight=0.9,
                       process_focal_crop_entropy_weight=0.3,
                       process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False,
                       process_multicrop=None,
                       process_multicrop_mindim=None, process_multicrop_maxdim=None,
                       process_multicrop_minarea=None,
                       process_multicrop_maxarea=None, process_multicrop_objective=None,
                       process_multicrop_threshold=None, progress_cb=None,
                       caption_wd_interrogator_name=None, caption_wd_threshold=0.35):
    caption_wd_interrogator = None

    try:
        if process_caption:
            shared.interrogator.load()

        if process_caption_deepbooru:
            deepbooru.model.start()

        if caption_wd_interrogator_name:
            if caption_wd_interrogator_name not in interrogators:
                raise OSError(f'cannot found interrogator:{caption_wd_interrogator_name}')
            caption_wd_interrogator = interrogators[caption_wd_interrogator_name]

        sub_dirs = list(os.scandir(process_src))

        for i, dirname in enumerate(sub_dirs):
            basename = os.path.basename(dirname)
            dst_dir = os.path.join(process_dst, basename)
            print(f'preprocess work with:{dirname}')
            if basename.startswith('.') or 'MACOSX' in basename:
                print('ignore macosx')
                continue

            def work_progress(p):
                task_progress = (i + 1) / len(sub_dirs) * p
                if callable(progress_cb):
                    progress_cb(task_progress)

            preprocess_work(dirname, dst_dir, process_width, process_height, preprocess_txt_action, process_keep_original_size,
                            process_flip, process_split, process_caption, process_caption_deepbooru, split_threshold,
                            overlap_ratio, process_focal_crop, process_focal_crop_face_weight,
                            process_focal_crop_entropy_weight, process_focal_crop_edges_weight,
                            process_focal_crop_debug, process_multicrop, process_multicrop_mindim,
                            process_multicrop_maxdim, process_multicrop_minarea, process_multicrop_maxarea,
                            process_multicrop_objective, process_multicrop_threshold, work_progress,
                            caption_wd_interrogator, caption_wd_threshold)
    finally:

        if process_caption:
            shared.interrogator.send_blip_to_ram()

        if process_caption_deepbooru:
            deepbooru.model.stop()

        if caption_wd_interrogator and isinstance(caption_wd_interrogator, WaifuDiffusionInterrogator):
            caption_wd_interrogator.unload()

