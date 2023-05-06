#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 6:07 PM
# @Author  : wangdongming
# @Site    : 
# @File    : typex.py
# @Software: Hifive
import os.path
import shutil
import typing
from enum import IntEnum
from collections import UserDict
from tools.encryptor import des_encrypt
from tools.image import compress_image
from PIL.PngImagePlugin import PngInfo
from filestorage import find_storage_classes_with_env
from tools.environment import get_file_storage_system_env, Env_BucketKey


class ModelType(IntEnum):
    Embedding = 1
    CheckPoint = 2
    Lora = 3


class OutImageType(IntEnum):
    Grid = 1
    Image = 2
    Script = 3


ModelLocation = {
    ModelType.Embedding: 'embendings',
    ModelType.CheckPoint: 'models/Stable-diffusion',
    ModelType.Lora: 'models/Lora'
}

UserModelLocation = {
    ModelType.Embedding: 'embendings',
    ModelType.CheckPoint: 'user-models/Stable-diffusion',
    ModelType.Lora: 'user-models/Lora'
}

S3ImageBucket = "xingzhe-sdplus"
S3ImagePath = "output/{uid}/{dir}/{name}"
S3Tmp = 'sd-tmp'


class ImageKeys(UserDict):

    def __init__(self, keys: typing.Sequence, low_keys: typing.Sequence):
        super(ImageKeys, self).__init__()
        self['high'] = keys or []
        self['low'] = low_keys or []

    def is_empty(self):
        return len(self['high']) == 0

    def __add__(self, ik: UserDict):
        high = self['high'] + ik['high']
        low = self['low'] + ik['low']
        return ImageKeys(high, low)


class ImageOutput:

    def __init__(self, image_type: OutImageType, local_output_dir: str):
        self.local_files = []
        self.image_type = image_type
        self.output_dir = local_output_dir
        os.makedirs(local_output_dir, exist_ok=True)

    def get_local_low_images(self):
        local_low_images = []
        for image_path in self.local_files:
            filename = os.path.basename(image_path)
            low_file = os.path.join(self.output_dir, 'low-' + filename)
            if not os.path.isfile(low_file):
                compress_image(image_path, low_file)
            local_low_images.append(low_file)
        return local_low_images

    def add_image(self, image: str):
        if os.path.isfile(image):
            self.local_files.append(image)
        else:
            raise OSError(f'cannot found image:{image}')

    def upload_keys(self, clean_upload_file: bool = True):
        file_storage_system_cls = find_storage_classes_with_env()
        file_storage_system = file_storage_system_cls()
        storage_env = get_file_storage_system_env()
        bucket = storage_env.get(Env_BucketKey) or S3ImageBucket
        low_files = self.get_local_low_images()
        # push s3
        if file_storage_system.name() != 'default':
            low_keys, keys = [], []
            for low_file in low_files:
                low_key = os.path.join(bucket, low_file)
                file_storage_system.upload(low_file, low_key)
                low_keys.append(low_key)

            for file_path in self.local_files:
                filename = os.path.basename(file_path)
                key = os.path.join(bucket, self.output_dir, filename)
                file_storage_system.upload(file_path, key)
                keys.append(key)

            if clean_upload_file:
                try:
                    shutil.rmtree(self.output_dir)
                except:
                    pass

            return ImageKeys(keys, low_keys)

        # local
        return ImageKeys(self.local_files, low_files)
