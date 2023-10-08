#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 9:54 AM
# @Author  : wangdongming
# @Site    : 
# @File    : handler.py
# @Software: Hifive
import abc
import random
import time
import typing
import traceback

import torch.cuda
from modules.shared import mem_mon as vram_mon
from worker.dumper import dumper
from loguru import logger
from modules.devices import torch_gc, get_cuda_device_string
from worker.k8s_health import write_healthy, system_exit
from worker.task import Task, TaskProgress, TaskStatus, TaskType


class TaskHandler:

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.enable = True

    def handle_task_type(self):
        return self.task_type

    @abc.abstractmethod
    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        raise NotImplementedError

    def _set_task_status(self, p: TaskProgress):
        logger.info(f">>> task:{p.task.desc()}, status:{p.status.name}, desc:{p.task_desc}")

    def do(self, task: Task, progress_callback=None):
        ok, msg = task.valid()
        if not ok:
            p = TaskProgress.new_failed(task, msg)
            self._set_task_status(p)
        else:
            try:
                p = TaskProgress.new_prepare(task, msg)
                self._set_task_status(p)
                for progress in self._exec(task):
                    self._set_task_status(progress)
                    if callable(progress_callback):
                        progress_callback(progress)

            except torch.cuda.OutOfMemoryError:
                torch_gc()
                time.sleep(15)
                logger.exception('CUDA out of memory')
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
                p = TaskProgress.new_failed(
                    task,
                    'CUDA out of memory',
                    f'CUDA out of memory and release, free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                self._set_task_status(p)
                progress_callback(p)
                system_exit(free, total)
            except Exception as ex:
                trace = traceback.format_exc()
                msg = str(ex)
                logger.exception('unhandle err')
                p = TaskProgress.new_failed(task, msg, trace)

                self._set_task_status(p)
                progress_callback(p)
                torch_gc()
                if 'BrokenPipeError' in str(ex):
                    pass

    def close(self):
        pass

    def set_failed(self, task: Task, desc: str):
        p = TaskProgress.new_failed(task, desc)
        self._set_task_status(p)

    def __call__(self, *args, **kwargs):
        return self.do(*args, **kwargs)


class DumpTaskHandler(TaskHandler, abc.ABC):
    
    def __init__(self, task_type: TaskType):
        super(DumpTaskHandler, self).__init__(task_type)

    def _set_task_status(self, p: TaskProgress):
        super()._set_task_status(p)
        dumper.dump_task_progress(p)

    def close(self):
        dumper.stop()