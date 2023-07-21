#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 9:38 PM
# @Author  : wangdongming
# @Site    : 
# @File    : k8s_health.py
# @Software: Hifive
import os


def write_healthy(status: bool):
    if status:
        with open("/var/healthy.txt", "w+") as f:
            v = "0" if status else '1'
            f.write(v)
    elif os.path.isfile("/var/healthy.txt"):
        os.remove("/var/healthy.txt")


