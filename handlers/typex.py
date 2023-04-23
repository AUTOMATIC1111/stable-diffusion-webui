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

S3ImageBucket = "xingzhe-sdplus"
S3ImagePath = "output/{uid}/{dir}/{name}"

