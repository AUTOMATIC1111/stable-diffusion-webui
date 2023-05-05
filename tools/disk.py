#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/5 4:24 PM
# @Author  : wangdongming
# @Site    : 
# @File    : disk.py
# @Software: Hifive
import ctypes
import os
import platform
import sys


def get_free_space_mb(folder: str) -> float:
    """
    Return folder/drive free space (in bytes)
    """
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024 / 1024
    else:
        st = os.statvfs(folder)
        return st.f_bavail * st.f_frsize / 1024 / 1024


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


def release_disk_with_free_mb(folder: str, expect: float):
    files = find_files_from_dir(folder)
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(x))
    free = get_free_space_mb('/')
    for i, f in enumerate(sorted_files):
        if free > expect:
            break
        try:
            fsize = os.path.getsize(f) / 1024 / 1024
            os.remove(f)
            free += fsize
        except:
            continue

