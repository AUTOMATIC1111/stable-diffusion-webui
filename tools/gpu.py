#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:35 PM
# @Author  : wangdongming
# @Site    : 
# @File    : gpu.py
# @Software: Hifive
import typing
import pynvml


class GpuInfo:

    instance = None

    def __init__(self):
        try:
            pynvml.nvmlInit()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            device_count = pynvml.nvmlDeviceGetCount()
            device_mem_total = []
            names = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                device_mem_total.append(meminfo.total)
                names.append(f"{name}({uuid})")
            pynvml.nvmlShutdown()
            self.driver = driver_version
            self.mem_total = [x / 1024 / 1024 for x in device_mem_total]  # MB
            self.names = names
        except:
            self.names = []
            self.mem_total = 0
            self.driver = ''

    def __new__(cls, *args, **kwargs):
        if cls.instance is not None:
            return cls.instance

        cls.instance = object.__new__(cls)
        return cls.instance

