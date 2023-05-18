#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 2:29 PM
# @Author  : wangdongming
# @Site    : 
# @File    : file.py
# @Software: Hifive
import shutil
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

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    # for f in zip_list:
    #     zip_file.extract(f, dst)
    #     right_file = f.encode('cp437').decode('utf-8')
    #     os.rename(f, right_file)
    zip_file.extractall(path=dst)
    # 判断时需需要重复解包  并且针对zipfile解包的中文乱码问题进行修正
    for f in zip_list:
        # 常见的有两种编码，使用异常处理语句
        try:
            new_zip_file = f.encode('cp437').decode('gbk')
        except:
            try:
                new_zip_file = f.encode('cp437').decode('utf-8')
            except:
                new_zip_file = None
        if not new_zip_file:
            print(f"cannot encode file name:{f}")
        else:
            try:
                os.rename(os.path.join(dst, f), os.path.join(dst, new_zip_file))
            except Exception as ex:
                print(f"cannot rename {os.path.join(dst, f)} to {os.path.join(dst, new_zip_file)}")
                pass


def zip_compress(src, dst):
    filelist = []
    if os.path.isfile(dst):
        os.remove(dst)

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