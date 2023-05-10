#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 2:35 PM
# @Author  : wangdongming
# @Site    : 
# @File    : pull_repo_res.py
# @Software: Hifive
# 拉取Codeformer weights等文件。
import os
import shutil

import requests


def mkdir(path):
    build_dirs = []

    def find_builds(_path):
        if not os.path.isdir(_path):
            build_dirs.append(_path)
            parent = os.path.dirname(_path)
            if parent:
                find_builds(parent)

    find_builds(path)
    while len(build_dirs) > 0:
        os.mkdir(build_dirs.pop())


def pull_code_former_weights():
    base_url = 'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/sd-web/'
    file_ids = {
        'codeformer.pth': 'CodeFormer/codeformer.pth',
        'yolov5l-face.pth': 'facelib/detection_Resnet50_Final.pth',
        'parsing_parsenet.pth': 'facelib/parsing_parsenet.pth'
    }
    cf_weight_dir = '/CodeFormer/weights/'
    repositories = 'repositories'
    repositorie_weights = 'repositories-weight'
    mkdir(repositories + cf_weight_dir)

    for name, fid in file_ids.items():
        url = base_url + repositories + cf_weight_dir + fid
        filepath = repositories + cf_weight_dir + fid
        nfs = repositorie_weights + cf_weight_dir + fid
        if os.path.exists(filepath):
            continue
        if os.path.isfile(nfs):
            shutil.copyfile(nfs, filepath)
            continue
        print(f'download repository weight from:{url}')
        resp = requests.get(url, timeout=10)
        if resp.ok:
            chunk_size = 512
            current = 0
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        current += chunk_size
        else:
            raise Exception(f"cannot download code former weight, url:{url}")


def try_find_cache_json_file():
    file_path = "models/cache.json"
    if os.path.isfile(file_path):
        shutil.copy(file_path, "cache.json")


def pull_clip_category():
    interrogate = 'interrogate'
    os.makedirs(interrogate, exist_ok=True)
    categories = ['artists.txt', 'flavors.txt', 'mediums.txt', 'movements.txt']
    for name in categories:
        if not os.path.isfile(os.path.join(interrogate, name)):
            url = 'https://das-pub.obs.ap-southeast-1.myhuaweicloud.com/sd-webui/resource/' + name
            print(f'>> download {name}')
            resp = requests.get(url, timeout=10)
            if resp.ok:
                with open(os.path.join(interrogate, name), "wb+") as f:
                    f.write(resp.content)


def pull_res():
    pull_clip_category()
    pull_code_former_weights()
    try_find_cache_json_file()
