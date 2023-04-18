#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 6:29 PM
# @Author  : wangdongming
# @Site    : 
# @File    : wrapper.py
# @Software: Hifive
import json
import time
from loguru import logger
from functools import wraps


class FuncExecTimeWrapper(object):

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            print('function {} spend:{} sec.'.format(func.__name__, t2-t1))
            return res
        return wrapper


class FuncResultLogWrapper:

    def __init__(self, msg: str):
        self.msg = msg

    def __call__(self, f):

        def wrapper(*args, **kwargs):
            r = f(*args, **kwargs)
            logger.info(self.msg + json.dumps(r))
            return r
        return wrapper

