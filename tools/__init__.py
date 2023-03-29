#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive
import json


def try_deserialize_json(json_str: str, default=None):
    try:
        v = json.loads(json_str)
        return v
    except :
        return default

