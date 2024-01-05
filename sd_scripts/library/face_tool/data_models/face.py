#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 6:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : face.py
# @Software: Hifive
import typing

from pydantic import BaseModel


class FaceQualityRequest(BaseModel):
    main_image_key: str
    data_keys: typing.List[str] = []


class ImgFaceQuality(BaseModel):

    file_name: str
    quality: int
    gender: int
    age: int