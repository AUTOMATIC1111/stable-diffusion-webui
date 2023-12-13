#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive
import json
import os.path
import shutil
import time


def try_deserialize_json(json_str: str, default=None):
    try:
        v = json.loads(json_str)
        return v
    except :
        return default


TempDir = "tmp"

os.makedirs(TempDir, exist_ok=True)


def safety_clean_tmp(exp=129600):
    if os.path.isdir(TempDir):
        files = [x for x in os.listdir(TempDir)]
        now = time.time()
        for f in files:
            full_path = os.path.join(TempDir, f)
            ctime = os.path.getctime(full_path)
            if exp > 0 and ctime + exp > now:
                continue

            try:
                print(f"remove file path:{full_path}")
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
            except Exception as e:
                print(f'cannot remove file:{e}, path:{full_path}')
    else:
        os.makedirs(TempDir, exist_ok=True)
