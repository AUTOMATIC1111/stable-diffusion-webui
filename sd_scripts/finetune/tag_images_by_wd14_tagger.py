import argparse
import csv
# import glob
import os
import re
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
# from tensorflow.keras.models import load_model
# from huggingface_hub import hf_hub_download
# import torch
from pathlib import Path

import sd_scripts.library.train_util as train_util

# from wd14 tagger
IMAGE_SIZE = 448

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

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

tag_escape_pattern = re.compile(r'([\\()])')

# def preprocess_image(image):
#     image = np.array(image)
#     image = image[:, :, ::-1]  # RGB->BGR

#     # pad to square
#     size = max(image.shape[0:2])
#     pad_x = size - image.shape[1]
#     pad_y = size - image.shape[0]
#     pad_l = pad_x // 2
#     pad_t = pad_y // 2
#     image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

#     interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
#     image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

#     image = image.astype(np.float32)
#     return image


# class ImageLoadingPrepDataset(torch.utils.data.Dataset):
#     def __init__(self, image_paths):
#         self.images = image_paths

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = str(self.images[idx])

#         try:
#             image = Image.open(img_path).convert("RGB")
#             image = preprocess_image(image)
#             tensor = torch.tensor(image)
#         except Exception as e:
#             print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
#             return None

#         return (tensor, img_path)


# def collate_fn_remove_corrupted(batch):
#     """Collate function that allows to remove corrupted examples in the
#     dataloader. It expects that the dataloader returns 'None' when that occurs.
#     The 'None's in the batch are removed.
#     """
#     # Filter out all the Nones (corrupted examples)
#     batch = list(filter(lambda x: x is not None, batch))
#     return batch

from typing import Tuple, List, Dict
class WaifuDiffusionInterrogator():
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
        self.model=None
        self.tags=None

    def load(self,path) -> None:
        from onnxruntime import InferenceSession
        import pandas as pd

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model_path = os.path.join(path,self.model_path)
        tags_path = os.path.join(path,self.tags_path)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded wd model from {model_path}')

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

        return unloaded
    
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:
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

wdInterrogator = WaifuDiffusionInterrogator("") 

def get_wd_tagger(train_data_dir="", # 训练数据路径
    model_dir="", # wd模型的地址
    caption_extension=".txt", # 存tag的文件类型
    general_threshold=0.35, # 为一般类别添加标签的置信阈值
    recursive=True, # 搜索子文件夹中的图像
    remove_underscore=True, # 将输出标记的下划线替换为空格
    undesired_tags="", # 不想要（想要去除）的Tag，以英文逗号隔开
    additional_tags="",):
    if wdInterrogator.model is None:
        wdInterrogator.load(model_dir)
    
    train_data_dir_path = Path(train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, recursive)
    for path in image_paths:
        try:
            image = Image.open(path)
        except:
            # just in case, user has mysterious file...
            print(f'${path} is not supported image type')
            continue
        
        _, tags = wdInterrogator.interrogate(image)
        # print(f"{path}")
        post_tags = wdInterrogator.postprocess_tags(tags=tags, threshold=general_threshold, additional_tags=additional_tags.split(","),
                                                    exclude_tags=undesired_tags.split(","),sort_by_alphabetical_order=False,
                                                    add_confident_as_weight=False,replace_underscore=remove_underscore,
                                                    replace_underscore_excludes=[],escape_tag=False)

        tag_text = ", ".join(post_tags)

        with open(os.path.splitext(path)[0] + caption_extension, "wt", encoding="utf-8") as f:
            f.write(tag_text + "\n")

    wdInterrogator.unload()

