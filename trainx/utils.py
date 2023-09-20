#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 2:54 PM
# @Author  : wangdongming
# @Site    : 
# @File    : utils.py
# @Software: Hifive
import hashlib
import os
import numpy as np
from PIL import Image
from enum import IntEnum
from loguru import logger
from worker.task_recv import Tmp
from datetime import datetime
from insightface.app import FaceAnalysis
from tools.environment import get_file_storage_system_env, Env_BucketKey, S3ImageBucket, S3Tmp, S3SDWEB
from filestorage import FileStorageCls, get_local_path, batch_download, http_down


class ModelType(IntEnum):
    Embedding = 1
    CheckPoint = 2
    Lora = 3


ModelLocation = {
    ModelType.Embedding: 'embeddings',
    ModelType.CheckPoint: 'models/Stable-diffusion',
    ModelType.Lora: 'models/Lora'
}

UserModelLocation = {
    ModelType.Embedding: 'embendings',
    ModelType.CheckPoint: 'user-models/Stable-diffusion',
    ModelType.Lora: 'user-models/Lora'
}


def calculate_sha256(filename, size=0):
    hash_sha256 = hashlib.sha256()
    blksize = size if size > 0 else 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_model_local_path(remoting_path: str, model_type: ModelType):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path
    # 判断user-models下
    os.makedirs(UserModelLocation[model_type], exist_ok=True)
    dst = os.path.join(UserModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        return dst

    os.makedirs(ModelLocation[model_type], exist_ok=True)
    dst = os.path.join(ModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        return dst

    dst = get_local_path(remoting_path, dst)
    if os.path.isfile(dst):
        # if model_type == ModelType.CheckPoint:
        #     checkpoint = CheckpointInfo(dst)
        #     checkpoint.register()
        return dst


def get_tmp_local_path(remoting_path: str, dir=None):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path

    dirname = Tmp if not dir else dir
    os.makedirs(dirname, exist_ok=True)
    dst = os.path.join(dirname, os.path.basename(remoting_path))
    return get_local_path(remoting_path, dst)


def upload_files(is_tmp, *files, dirname=None):
    keys = []
    if files:
        date = datetime.today().strftime('%Y/%m/%d')
        storage_env = get_file_storage_system_env()
        bucket = storage_env.get(Env_BucketKey) or S3ImageBucket
        file_storage_system = FileStorageCls()
        relative = S3Tmp if is_tmp else S3SDWEB
        if dirname:
            relative = os.path.join(relative, dirname)

        for f in files:
            name = os.path.basename(f)
            key = os.path.join(bucket, relative, date, name)
            file_storage_system.upload(f, key)
            keys.append(key)
    return keys


def detect_image_face(*images):
    def download_models():
        buffalo_l = os.path.join('models', 'buffalo_l')
        os.makedirs(buffalo_l, exist_ok=True)
        model_names = [
            '1k3d68.onnx',
            '2d106det.onnx',
            'det_10g.onnx',
            'genderage.onnx',
            'w600k_r50.onnx',
        ]
        base_url = 'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/face-analysis/buffalo_l/'

        for m in model_names:
            local = os.path.join(buffalo_l, m)
            if not os.path.isfile(local):
                logger.info(f"download buffalo_l: {local}...")
                http_down(base_url + m, local)

    download_models()
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='.')
    app.prepare(ctx_id=0, det_size=(640, 640))
    for img_path in images:
        img = np.array(Image.open(img_path).convert("RGB"))
        faces = app.get(img)
        basename = os.path.basename(img_path)
        if not faces:
            yield (basename, None)
        else:
            yield (basename, faces[0])
