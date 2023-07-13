#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 2:54 PM
# @Author  : wangdongming
# @Site    : 
# @File    : utils.py
# @Software: Hifive
import os
from enum import IntEnum
from worker.task_recv import Tmp
from datetime import datetime
from tools.environment import get_file_storage_system_env, Env_BucketKey, S3ImageBucket, S3Tmp, S3SDWEB
from filestorage import FileStorageCls, get_local_path, batch_download


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
        if model_type == ModelType.CheckPoint:
            checkpoint = CheckpointInfo(dst)
            checkpoint.register()
        return dst


def get_tmp_local_path(remoting_path: str):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path

    os.makedirs(Tmp, exist_ok=True)
    dst = os.path.join(Tmp, os.path.basename(remoting_path))
    return get_local_path(remoting_path, dst)


def upload_files(is_tmp, *files):
    keys = []
    if files:
        date = datetime.today().strftime('%Y/%m/%d')
        storage_env = get_file_storage_system_env()
        bucket = storage_env.get(Env_BucketKey) or S3ImageBucket
        file_storage_system = FileStorageCls()
        relative = S3Tmp if is_tmp else S3SDWEB

        for f in files:
            name = os.path.basename(f)
            key = os.path.join(bucket, relative, date, name)
            file_storage_system.upload(f, key)
            keys.append(key)
    return keys