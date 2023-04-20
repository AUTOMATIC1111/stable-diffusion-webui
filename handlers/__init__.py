#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : __init__.py.py
# @Software: Hifive

from worker.task import TaskHandler
from tools.reflection import find_classes
from handlers.img2img import Img2ImgTaskHandler
from handlers.txt2img import Txt2ImgTaskHandler


def get_task_handlers():
    for cls in find_classes("handlers"):
        if issubclass(cls, TaskHandler) and cls != TaskHandler:
            yield cls()


def_task_handlers = [
    Img2ImgTaskHandler(),
    Txt2ImgTaskHandler()
]
