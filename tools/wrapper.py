#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 6:29 PM
# @Author  : wangdongming
# @Site    : 
# @File    : wrapper.py
# @Software: Hifive

import time


class FuncExecTimeWrapper(object):

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            print('function {} spend:{} sec.'.format(func.__name__, t2-t1))
            return res
        return wrapper


