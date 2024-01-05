#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 10:55 AM
# @Author  : wangdongming
# @Site    : 
# @File    : face.py
# @Software: Hifive
import os
import shutil

from data_models.types import *
from data_models.face import FaceQualityRequest
from data_models.resp import BaseResponse, FastHandlerErrorDecorator
from super.face import insightface_main_face, insightface_face_recognition
from filestorage import get_local_path, batch_download, get_tmp_path, FileStorageCls
from fastapi import APIRouter

router = APIRouter(
    prefix=f'{V1}/face',
    tags=['face']
)


@router.post(f'/reco/main')
@FastHandlerErrorDecorator
def insight_main_face(r: FaceQualityRequest):
    tmp = next(get_tmp_path(r.main_image_key))
    get_local_path(r.main_image_key, tmp, FileStorageCls)
    if not os.path.isfile(tmp):
        raise OSError(f'cannot download {r.main_image_key}')
    faces, quality = insightface_main_face(tmp)
    os.remove(tmp)

    return [quality]


@router.post(f'/reco')
@FastHandlerErrorDecorator
def insight_train_image_face_quality(r: FaceQualityRequest) -> BaseResponse:
    keys = [r.main_image_key]
    keys.extend(r.data_keys)

    remoting_loc_pairs = []
    for i, local in enumerate(get_tmp_path(*r.data_keys)):
        remoting_loc_pairs.append((keys[i], local))

    batch_download(remoting_loc_pairs, FileStorageCls)

    main_image_path, data_dir = '', ''
    for pairs in remoting_loc_pairs:
        local = pairs[-1]
        key = pairs[0]
        if not os.path.isfile(local):
            raise OSError(f'cannot download {key}')
        if key == r.main_image_key:
            main_image_path = local
        if not data_dir:
            data_dir = os.path.dirname(local)
    r = insightface_face_recognition(main_image_path, data_dir)
    shutil.rmtree(data_dir)
    return r


@router.get('/')
@FastHandlerErrorDecorator
def index():
    return 'hello face'


