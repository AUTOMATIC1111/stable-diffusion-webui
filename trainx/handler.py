#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 3:47 PM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
from enum import IntEnum
from worker.handler import DumpTaskHandler
from worker.task import Task, TaskType, TaskProgress, TrainEpoch
from .preprocess import exec_preprocess_task
from .lora import exec_train_lora_task


class TrainTaskMinorType(IntEnum):

    Preprocess = 1
    Lora = 2


class TrainTaskHandler(DumpTaskHandler):

    def __init__(self):
        super(TrainTaskHandler, self).__init__(TaskType.Train)

    def _exec(self, task: Task):
        if task.minor_type == TrainTaskMinorType.Preprocess:
            yield from exec_preprocess_task(task)
        elif task.minor_type == TrainTaskMinorType.Lora:
            def progress_callback(epoch, loss_total, num_train_epochs):
                progress = epoch / num_train_epochs * 100 * 0.9
                p = TaskProgress.new_running(task, 'running', progress)
                p.train.add_epoch_log(TrainEpoch(epoch, loss_total))
                self._set_task_status(p)

            yield from exec_train_lora_task(task, progress_callback)
