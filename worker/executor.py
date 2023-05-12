#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 2:17 PM
# @Author  : wangdongming
# @Site    : 
# @File    : executor.py
# @Software: Hifive
import time
import typing
from queue import Queue
from loguru import logger
from worker.task import Task
from worker.handler import TaskHandler
from worker.task_recv import TaskReceiver, TaskTimeout
from threading import Thread, Condition, Lock
from tools.model_hist import CkptLoadRecorder


class TaskExecutor(Thread):

    def __init__(self, ckpt_recorder: CkptLoadRecorder, timeout=0, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None, train_only=False):
        self.receiver = TaskReceiver(ckpt_recorder, train_only)
        self.recorder = ckpt_recorder
        self._handlers = {}
        self.timeout = timeout if timeout > 0 else TaskTimeout
        self.stop = False
        self.mutex = Lock()
        self.not_busy = Condition(self.mutex)
        self.queue = Queue(1)  # 也可直接使用变量进行消息传递。。
        name = name or 'task-executor'
        super(TaskExecutor, self).__init__(group, target, name, args, kwargs, daemon=daemon)

    def _close(self):
        for h in self._handlers:
            h.close()

    def stop(self):
        self.stop = True

    def add_handler(self, *handlers: TaskHandler):
        for handler in handlers:
            self._handlers[handler.task_type] = handler

    def get_handler(self, task: Task) -> typing.Optional[TaskHandler]:
        return self._handlers.get(task.task_type)

    def error_handler(self, task: Task, ex: Exception):
        logger.error(f'exec task failed: {task.desc()}, ex: {ex}')

    def exec_task(self):
        handlers = [x.name for x in self._handlers.keys()]
        logger.info(f"executor start with:{','.join(handlers)}")
        while self.is_alive():
            task = self.queue.get()
            logger.info(f"====>>> receive task:{task.desc()}")
            logger.info(f"====>>> model history:{self.recorder.history()}")

            handler = self.get_handler(task)
            if not handler:
                self.error_handler(task, Exception('can not found task handler'))
            if task.create_at > 0 and time.time() - task.create_at > self.timeout:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.create_at))
                handler.set_failed(task, f'task time out(task create time:{create_time}, now:{now})')
                continue
            handler(task)
            self.not_busy.notify()
        logger.info('executor stopping...')
        self._close()

    def _get_task(self):
        while self.is_alive():
            with self.not_busy:
                if not self.queue.full():
                    for task in self.receiver.task_iter():
                        self.queue.put(task)
                else:
                    self.not_busy.wait()

    def run(self) -> None:
        self._get_task()

    def start(self) -> None:
        super(TaskExecutor, self).start()
        self.exec_task()

