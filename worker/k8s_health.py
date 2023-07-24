#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 9:38 PM
# @Author  : wangdongming
# @Site    : 
# @File    : k8s_health.py
# @Software: Hifive
import os
from loguru import logger


def write_healthy(status: bool):
    if status:
        with open("/var/healthy.txt", "w+") as f:
            v = "0" if status else '1'
            f.write(v)
    elif os.path.isfile("/var/healthy.txt"):
        os.remove("/var/healthy.txt")


def system_exit(free, threshold=8):
    if free / 2 ** 30 < threshold:
        logger.info("CUDA out of memory, quit...")
        # kill process
        from ctypes import CDLL
        from ctypes.util import find_library

        write_healthy(False)

        libc = CDLL(find_library("libc"))
        libc.exit(1)

