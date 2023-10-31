#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 9:38 PM
# @Author  : wangdongming
# @Site    : 
# @File    : k8s_health.py
# @Software: Hifive
import os
import time
import pstuil
import psutil
from loguru import logger
from ctypes import CDLL
from ctypes.util import find_library


def write_healthy(status: bool):
    if status:
        try:
            with open("/var/healthy.txt", "w+") as f:
                v = "0" if status else '1'
                f.write(v)
        except Exception as e:
            print(f"cannot write file /var/healthy.txt:{e}")
    elif os.path.isfile("/var/healthy.txt"):
        os.remove("/var/healthy.txt")


def _exit():
    # for restart k8s pod
    write_healthy(False)
    time.sleep(1)

    # kill process
    libc = CDLL(find_library("libc"))
    libc.exit(1)


def system_exit(free, total, threshold=0.2, coercive=False):
    gpu_oom = free < threshold * total and free / 2 ** 30 < 3

    if gpu_oom or coercive:
        if gpu_oom:
            logger.info(f"CUDA out of memory({free}/{total}), quit...")
        else:
            logger.info("kill current process.")
        # for restart k8s pod
        _exit()

    else:
        mem = psutil.virtual_memory()
        total = int(mem.total / 1024 / 1024 / 1024)
        used = int(mem.used / 1024 / 1024 / 1024)
        if used > total * 0.8:
            logger.info(f"out of memory:{used}/{total}(GB), kill current process...")
            _exit()

