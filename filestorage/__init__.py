#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 4:01 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive
import os
import typing

from loguru import logger
from filestorage.storage import FileStorage, PrivatizationFileStorage
from tools.reflection import find_classes
from urllib.parse import urlparse
from tools.environment import Env_EndponitKey


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


def get_local_path(remoting, local, storage_cls=None, progress_callback=None, with_locker=False, locker_exp=None):
    if os.path.isfile(local):
        return local
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        if not with_locker:
            return s.download(remoting, local, progress_callback)
        else:
            return s.lock_download(remoting, local, progress_callback, locker_exp)


def batch_download(remoting_loc_pairs: typing.Sequence[typing.Tuple[str, str]], storage_cls=None, with_locker=False):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.multi_download(remoting_loc_pairs, with_locker)


def push_local_path(remoting, local, storage_cls=None):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.upload(local, remoting)


def signature_url(remoting, storage_cls=None):
    storage_cls = storage_cls or find_storage_classes_with_env()
    with storage_cls() as s:
        return s.preview_url(remoting)


def http_down(remoting, local):
    f = PrivatizationFileStorage()
    f.download(remoting, local)


FileStorageCls = find_storage_classes_with_env()



