#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 9:43 AM
# @Author  : wangdongming
# @Site    : 
# @File    : host.py
# @Software: Hifive
import socket


def get_host_ip():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        if s:
            s.close()
    return ip
