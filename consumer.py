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
from trainx import train_task_handlers


def run_executor(ckpt_recorder: CkptLoadRecorder, *handlers: TaskHandler, train_only=False):
    handlers = handlers or get_task_handlers()
    handlers = list(handlers)
    handlers.extend(train_task_handlers)
    executor = TaskExecutor(ckpt_recorder, train_only=train_only)
    executor.add_handler(*handlers)
    executor.start()
    return executor
