#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 2:29 PM
# @Author  : wangdongming
# @Site    : 
# @File    : file.py
# @Software: Hifive
import zipfile
import os

from os.path import join, getsize


def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files])
    return size


def zip_uncompress(src, dst):
    zip_file = zipfile.ZipFile(src)
    zip_list = zip_file.namelist()

    try:
        for f in zip_list:
            zip_file.extract(f, dst)
    finally:
        zip_file.close()


def zip_compress(src, dst):
    filelist = []
    if os.path.isfile(src):
        filelist.append(src)
    else:
        for root, dirs, files in os.walk(src):
            for name in files:
                filelist.append(os.path.join(root, name))

    zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    for tar in filelist:
        arcname = tar[len(dst):]
        zf.write(tar, arcname)
    zf.close()


def find_files_from_dir(directory, *args):
    extensions_ = [ex.lstrip('.').upper() for ex in args]
    for fn in os.listdir(directory):
        full_path = os.path.join(directory, fn)
        if os.path.isfile(full_path):
            if extensions_:
                _, ex = os.path.splitext(full_path)
                if ex.lstrip('.').upper() in extensions_:
                    yield full_path
            else:
                yield full_path
        elif os.path.isdir(full_path):
            for f in find_files_from_dir(full_path, *extensions_):
                yield f