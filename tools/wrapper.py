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
from functools import lru_cache, wraps
from datetime import datetime, timedelta


class FuncExecTimeWrapper(object):

    def __call__(self, func):
        @wraps(func)
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
        @wraps(f)
        def wrapper(*args, **kwargs):
            r = f(*args, **kwargs)
            logger.info(self.msg + json.dumps(r))
            return r
        return wrapper


def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache



