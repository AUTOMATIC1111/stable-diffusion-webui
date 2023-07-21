#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 9:38 PM
# @Author  : wangdongming
# @Site    : 
# @File    : k8s_health.py
# @Software: Hifive

def write_healthy(status: bool):
    with open("/var/healthy.txt", "w+") as f:
        v = "0" if status else '1'
        f.write(v)


