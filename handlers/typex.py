#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 6:07 PM
# @Author  : wangdongming
# @Site    : 
# @File    : typex.py
# @Software: Hifive
from enum import IntEnum


class ModelType(IntEnum):
    Embedding = 1
    CheckPoint = 2
    Lora = 3


ModelLocation = {
    ModelType.Embedding: 'embendings',
    ModelType.CheckPoint: 'models/Stable-diffusion',
    ModelType.Lora: 'models/Lora'
}

Tmp = 'tmp'
S3ImageBucket = "sd-webui-res"
S3ImagePath = "{uid}/output/{name}"

