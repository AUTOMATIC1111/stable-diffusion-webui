#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 3:47 PM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
from enum import IntEnum
from worker.handler import DumpTaskHandler
from worker.task import Task, TaskType
from .preprocess import exec_preprocess_task
from .lora import exec_train_lora_task, start_train_process
from modules.devices import torch_gc


class TrainTaskMinorType(IntEnum):

    Preprocess = 1
    Lora = 2




class TrainTaskHandler(DumpTaskHandler):

    def __init__(self):
        super(TrainTaskHandler, self).__init__(TaskType.Train)

    def _exec(self, task: Task):
        torch_gc()
        if task.minor_type == TrainTaskMinorType.Preprocess:
            yield from exec_preprocess_task(task)
        elif task.minor_type == TrainTaskMinorType.Lora:
            yield from exec_train_lora_task(task, self._set_task_status)
