#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 1:41 PM
# @Author  : wangdongming
# @Site    : 
# @File    : consumer.py
# @Software: Hifive
from tools.model_hist import CkptLoadRecorder
from worker.executor import TaskExecutor, TaskHandler
from handlers import get_task_handlers


def run_executor(ckpt_recorder: CkptLoadRecorder, *handlers: TaskHandler):
    handlers = handlers or get_task_handlers()
    executor = TaskExecutor(ckpt_recorder)
    executor.add_handler(*handlers)
    executor.start()
    return executor
