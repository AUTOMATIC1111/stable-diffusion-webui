#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:01 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive
import os
import typing
import uuid
import time
import shutil

from loguru import logger
from library.face_tool.filestorage.storage import FileStorage
from library.face_tool.tools.reflection import find_classes
from urllib.parse import urlparse
from library.face_tool.tools.environment import Env_EndponitKey

TmpDir = 'tmp'
os.makedirs(TmpDir, exist_ok=True)


def clean_tmp_dir():
    expired_days = 1
    if os.path.isdir(TmpDir):
        now = time.time()
        files = [x for x in os.listdir(TmpDir)]
        for fn in files:
            if not os.path.isfile(fn):
                continue
            mtime = os.path.getmtime(fn)
            if now > mtime + expired_days * 24 * 3600:
                try:
                    if os.path.isdir(fn):
                        shutil.rmtree(fn)
                    else:
                        os.remove(fn)
                except Exception:
                    logger.exception('cannot remove file!!')


def find_storage_classes_with_env():
    endpoint = os.getenv(Env_EndponitKey)
    if not endpoint:
        logger.warning('[storage] > cannot found storage system config, use local file storage system!!!')
    storages = {}
    domain = get_domain_from_endpoint(endpoint)
    for cls in find_classes('filestorage'):
        if issubclass(cls, FileStorage) and cls != FileStorage:
            ins = cls()
            if ins.name():
                storages[ins.name()] = cls
    if domain and domain in storages:
        return storages[domain]

    return storages.get('default')


def get_domain_from_endpoint(endpoint):
    if not endpoint:
        return 'default'
    domain = urlparse(endpoint).netloc or endpoint
    array = domain.split('.')
    if len(array) > 2 and array[-1].lower() == 'com':
        domain = array[-2]
    return domain.lower()


def get_local_path(remoting, local, storage_cls=None, progress_callback=None):
    if os.path.isfile(local):
        return local
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        r = s.download(remoting, local, progress_callback)


def batch_download(remoting_loc_pairs: typing.Sequence[typing.Tuple[str, str]], storage_cls=None):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.multi_download(remoting_loc_pairs)


def get_tmp_path(*remoting_keys, sub=None):
    relative = os.path.join(TmpDir, sub or str(uuid.uuid4()))
    os.makedirs(relative, exist_ok=True)
    for k in remoting_keys:

        yield os.path.join(relative, os.path.basename(k))


def push_local_path(remoting, local, storage_cls=None):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.upload(local, remoting)


def signature_url(remoting, storage_cls=None):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.preview_url(remoting)


FileStorageCls = find_storage_classes_with_env()



