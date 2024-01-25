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
import time
import typing

from loguru import logger


def get_free_space_mb(folder: str) -> float:
    """
    Return folder/drive free space (in bytes)
    """
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
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


def release_disk_with_free_mb(folder: str, expect_free: float, on_removing: typing.Callable = None):
    if not os.path.isdir(folder):
        return
    files = find_files_from_dir(folder)
    sorted_files = sorted(files, key=lambda x: os.path.getatime(x))
    free = 0
    for i, f in enumerate(sorted_files):
        if free > expect_free:
            break
        try:
            if on_removing and not on_removing(f):
                continue

            fsize = os.path.getsize(f) / 1024 / 1024
            os.remove(f)
            free += fsize
        except:
            continue


def tidy_model_caches(models_dir, expire_days: int = 14, persist_model_hashes: typing.Sequence = None):
    ckpt_dir = os.path.join(models_dir, 'Stable-diffusion')
    lora_dir = os.path.join(models_dir, 'Lora')
    lycor_dir = os.path.join(models_dir, 'LyCORIS')

    persist_model_hashes = persist_model_hashes or []
    persist_model_hash_map = {}
    for item in persist_model_hashes:
        key = item.decode() if isinstance(item, bytes) else item
        persist_model_hash_map[key] = 1
    persist_model_hash_map.update({'v1-5-pruned-emaonly': 1})

    def clean_timeout_files(expire_days):
        dirnames = [
            (ckpt_dir, 1),
            (lora_dir, 0.5),
            (lycor_dir, 0.5)
        ]
        now = time.time()
        interval = expire_days * 24 * 3600
        logger.debug("start clean disk...")
        for dir, bias in dirnames:
            if not os.path.isdir(dir):
                continue
            for f in os.listdir(dir):
                full = os.path.join(dir, f)
                if os.path.isfile(full):
                    try:
                        atime = os.path.getatime(full)
                        if now - atime < interval * bias:
                            continue
                        logger.warning(f'[WARN] file:{full}, atime expired!!!!')
                        os.remove(full)
                    except:
                        logger.exception(f'cannot remove file:{full}')

    def on_remove(x):
        basename, _ = os.path.splitext(os.path.basename(x))
        ok = basename not in persist_model_hash_map
        if not ok:
            print(f"[clean model] dont remove model:{x}")
        return ok

    if platform.system() == 'Windows':
        clean_timeout_files(expire_days)

    else:
        vfs = os.statvfs(models_dir)  # models可能是一个挂载点
        used = vfs.f_bsize * (vfs.f_blocks - vfs.f_bfree) / 1024
        avail = vfs.f_bsize * vfs.f_bavail / 1024
        used_percent = float(used) / float(used + avail)
        if used_percent > 0.85:
            # models可能是一个挂载点
            free = ((used + avail) * 0.2) // 1024
            release_disk_with_free_mb(ckpt_dir, free, on_remove)
            free = ((used + avail) * 0.1) // 1024
            release_disk_with_free_mb(lora_dir, free, on_remove)
            release_disk_with_free_mb(lycor_dir, free, on_remove)
        elif used_percent > 0.5:
            expire_days = expire_days // 2 if expire_days > 2 else expire_days
            clean_timeout_files(expire_days)
        else:
            clean_timeout_files(expire_days)
        vfs = os.statvfs(models_dir)
        free_mb = used - vfs.f_bsize * (vfs.f_blocks - vfs.f_bfree) / 1024
        logger.debug(f'>> release disk space:{free_mb / 1024} MB')
