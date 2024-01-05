#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 1:51 PM
# @Author  : wangdongming
# @Site    : 
# @File    : timer.py
# @Software: Hifive
import typing

from apscheduler.schedulers.background import BackgroundScheduler


class Timer:

    def __init__(self):
        self.timer = BackgroundScheduler()

        self.timer.start()

    def add_job(self, handler: typing.Callable, seconds: int, args=None, kwargs=None):
        self.timer.add_job(handler, 'interval', seconds=seconds, args=args, kwargs=kwargs)

    def stop(self):
        self.timer.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


