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
from collections import UserDict, defaultdict

import requests

from loguru import logger
from tools.image import compress_image
from PIL.PngImagePlugin import PngInfo
from filestorage import find_storage_classes_with_env, signature_url
from tools.environment import S3SDWEB, S3ImageBucket, Env_DtAppKey
from tools.processor import MultiThreadWorker


class ModelType(IntEnum):
    Unknown = -1
    CheckPoint = 1
    Lora = 2
    Embedding = 3
    LyCORIS = 4
    VAE = 5


class OutImageType(IntEnum):
    Grid = 1
    Image = 2
    Script = 3


ModelLocation = {
    ModelType.Embedding: 'embeddings',
    ModelType.CheckPoint: 'models/Stable-diffusion',
    ModelType.Lora: 'models/Lora',
    ModelType.LyCORIS: 'models/LyCORIS'
}

UserModelLocation = {
    ModelType.Embedding: 'embendings',
    ModelType.CheckPoint: 'user-models/Stable-diffusion',
    ModelType.Lora: 'user-models/Lora',
    ModelType.LyCORIS: 'user-models/LyCORIS'
}

ForbiddenCoverKey = 'sd-web/resources/403_blur.png'


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

    def sorted_keys(self, keys: typing.Sequence, forbidden_keys: typing.Mapping[str, typing.Any] = None,
                    has_grid: bool = False):
        def sort_file(p: str):
            basename, _ = os.path.splitext(os.path.basename(p))

            if '-' not in basename:
                return -1
            seg = basename.split('-')[-1]
            if seg == 'last':
                return 10000000
            try:
                x = int(seg)
            except:
                x = 0
            return x

        if not keys:
            return []

        sorted_keys = sorted(keys, key=sort_file)
        if forbidden_keys and sorted_keys:
            for i in range(len(sorted_keys)):
                key = sorted_keys[i]
                if key in forbidden_keys:
                    sorted_keys[i] = ForbiddenCoverKey
            if has_grid:
                sorted_keys[0] = ForbiddenCoverKey

        return sorted_keys

    def to_dict(self, forbidden_keys: typing.Mapping[str, typing.Any] = None, has_grid: bool = False):
        self['high'] = self.sorted_keys(self['high'], forbidden_keys, has_grid)
        self['low'] = self.sorted_keys(self['low'], forbidden_keys, has_grid)
        return dict(self)


def get_upload_image_key(file_storage_system, file: str, key: str, key_outs: typing.List[str]):
    r = file_storage_system.upload(file, key)
    if r:
        key_outs.append(r)


class ImageOutput:

    def __init__(self, image_type: OutImageType, local_output_dir: str):
        self.local_files = []
        self.image_type = image_type
        self.output_dir = local_output_dir
        os.makedirs(local_output_dir, exist_ok=True)

    def get_local_low_images(self):
        local_low_images, compress_images = [], []
        for image_path in self.local_files:
            filename = os.path.basename(image_path)
            _, ex = os.path.splitext(filename)
            # 不转GIF
            if ex.lower() == ".gif":
                continue
            low_file = os.path.join(self.output_dir, 'low-' + filename)
            if not os.path.isfile(low_file):
                compress_images.append((image_path, low_file))
                # compress_image(image_path, low_file)
            local_low_images.append(low_file)
        if compress_images:
            worker = MultiThreadWorker(compress_images, compress_image, 4)
            worker.run()
            for image_path in local_low_images:
                if not image_path:
                    raise OSError(f'cannot found low image:{image_path}')

        return local_low_images

    def add_image(self, image: str):
        if os.path.isfile(image):
            self.local_files.append(image)
        else:
            raise OSError(f'cannot found image:{image}')

    def upload_keys(self, clean_upload_file: bool = True):
        file_storage_system_cls = find_storage_classes_with_env()
        file_storage_system = file_storage_system_cls()

        low_files = self.get_local_low_images()
        # push s3
        if file_storage_system.name() != 'default':
            low_keys, keys = [], []
            for low_file in low_files:
                relative_path = low_file
                if S3SDWEB not in low_file:
                    relative_path = os.path.join(S3SDWEB, low_file)
                low_key = relative_path
                file_storage_system.upload(low_file, low_key)
                low_keys.append(low_key)

            relative_path = self.output_dir
            if S3SDWEB not in self.output_dir:
                relative_path = os.path.join(S3SDWEB, self.output_dir)
            for file_path in self.local_files:
                filename = os.path.basename(file_path)
                key = os.path.join(relative_path, filename)
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

    def multi_upload_keys(self, clean_upload_file: bool = True):
        file_storage_system_cls = find_storage_classes_with_env()
        file_storage_system = file_storage_system_cls()
        low_files = self.get_local_low_images()

        if file_storage_system.name() != 'default':
            low_keys, high_keys, worker_args = [], [], []
            for low_file in low_files:
                relative_path = low_file
                if S3SDWEB not in low_file:
                    relative_path = os.path.join(S3SDWEB, low_file)
                low_key = relative_path
                worker_args.append((file_storage_system, low_file, low_key, low_keys))

            relative_path = self.output_dir
            if S3SDWEB not in self.output_dir:
                relative_path = os.path.join(S3SDWEB, self.output_dir)
            for file_path in self.local_files:
                filename = os.path.basename(file_path)
                key = os.path.join(relative_path, filename)
                # file_storage_system.upload(file_path, key)
                worker_args.append((file_storage_system, file_path, key, high_keys))
            if worker_args:
                worker = MultiThreadWorker(worker_args, get_upload_image_key, 4)
                worker.run()

                if clean_upload_file:
                    try:
                        shutil.rmtree(self.output_dir)
                    except:
                        pass
                if len(high_keys) != len(low_keys):
                    raise OSError("upload image failed.")

                return ImageKeys(high_keys, low_keys)

            # local
        return ImageKeys(self.local_files, low_files)

    def inspect(self, images: ImageKeys, forbidden_review=False):
        api = 'https://diting.xingzheai.cn/api/v1.0/image/inspect'
        urls = {}
        url_inspect_map = {}

        for image in images['low']:
            try:
                url = signature_url(image)
                urls[image] = {
                    'url': url,
                    'data_id': 'tushuashua',
                    'text_inspect': 0,
                }
                url_inspect_map[url] = {
                    'key': image
                }
            except Exception as e:
                logger.exception('signature_url failed')
                continue

        forbidden_keys = defaultdict(int)
        try:
            if urls:
                dt_appkey = os.getenv(Env_DtAppKey, 'QX9HNRAYMFLBIG7T')

                resp = requests.post(api, json={
                    'token': dt_appkey,
                    'scenes': ["politics", "porn"],
                    'tasks': list(urls.values())
                }, timeout=5)

                logger.info(f"request {api}, app key:{dt_appkey}")
                if resp:
                    data = resp.json()
                    logger.debug(f"response {resp.text}")
                    if data.get('code', 0) == 200:
                        results = data['results']
                        for image_item in results:
                            img_res = image_item['result']
                            url = image_item['url']
                            for scene in img_res:
                                if scene['scene'] == 'porn' and not forbidden_review:
                                    forbidden_flag = scene.get('suggestion', 'pass') == 'block'
                                else:
                                    forbidden_flag = scene.get('suggestion', 'pass') != 'pass'
                                if forbidden_flag:
                                    if url in url_inspect_map:
                                        low_key = url_inspect_map[url]['key']
                                        dirname = os.path.dirname(low_key)
                                        basename = os.path.basename(low_key)
                                        high_key = os.path.join(dirname, basename.replace('low-', ''))
                                        forbidden_keys[low_key] = 1
                                        forbidden_keys[high_key] = 1
                                        logger.info(f"forbidden: {low_key}")
        except Exception as e:
            logger.exception(f'request {api} failed')
            pass

        # default_cover = ForbiddenCoverKey

        # for i, low_key in enumerate(images['low']):
        #     if low_key in forbidden_keys:
        #         images['low'][i] = default_cover
        #
        # for i, high_key in enumerate(images['high']):
        #     dirname = os.path.dirname(high_key)
        #     basename = os.path.basename(high_key)
        #     low_key = os.path.join(dirname, f'low-{basename}')
        #     if low_key in forbidden_keys:
        #         images['high'][i] = default_cover
        return forbidden_keys
