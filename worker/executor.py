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
from worker.task import Task, TaskProgress
from datetime import datetime
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
        self.__stop = False
        self.mutex = Lock()
        self.not_busy = Condition(self.mutex)
        self.queue = Queue(1)  # 也可直接使用变量进行消息传递。。
        name = name or 'task-executor'
        if train_only:
            logger.info("[executor] >>> run on train mode.")
        super(TaskExecutor, self).__init__(group, target, name, args, kwargs, daemon=daemon)

    def _close(self):
        for h in self._handlers:
            h.close()
        self.receiver.close()

    def stop(self):
        self.__stop = True

    def add_handler(self, *handlers: TaskHandler):
        for handler in handlers:
            self._handlers[handler.task_type] = handler

    def get_handler(self, task: Task) -> typing.Optional[TaskHandler]:
        return self._handlers.get(task.task_type)

    def error_handler(self, task: Task, ex: Exception):
        logger.error(f'exec task failed: {task.desc()}, ex: {ex}')

    def nofity(self):
        if getattr(self.not_busy, "value", 0) == 0:
            return

        with self.not_busy:
            self.not_busy.notify()
            logger.debug("notify receiver ")
            setattr(self.not_busy, "value", 0)

    def task_progress(self, p: TaskProgress):
        if p.pre_task_completed():
            self.nofity()

    def exec_task(self):
        handlers = [x.name for x in self._handlers.keys()]
        logger.info(f"executor start with:{','.join(handlers)}")
        while not self.__stop:
            try:
                task = self.queue.get()
                logger.info(f"====>>> receive task:{task.desc()}")
                logger.info(f"====>>> model history:{self.recorder.history()}")

                handler = self.get_handler(task)
                if not handler:
                    self.error_handler(task, Exception('can not found task handler'))
                # 判断TASK超时。
                if self._is_timeout(task):
                    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.create_at))
                    handler.set_failed(task, f'task time out(task create time:{create_time}, now:{now})')
                    continue
                handler(task, progress_callback=self.task_progress)
            except Exception:
                logger.exception("executor err")
                self.nofity()

        logger.info('executor stopping...')
        self._close()

    def _get_task(self):
        while self.is_alive():
            with self.not_busy:
                if not self.queue.full():
                    for task in self.receiver.task_iter():
                        logger.info(f"====>>> preload task:{task.id}")
                        self.queue.put(task)
                        if isinstance(task, Task):
                            logger.debug(f"====>>> waiting task:{task.id}, stop receive.")
                            setattr(self.not_busy, "value", 1)
                            self.not_busy.wait()
                            logger.debug(f"====>>> waiting task:{task.id}, begin receive.")
                else:
                    self.not_busy.wait()
        logger.info("=======> task receiver quit!!!!!!")

    def _is_timeout(self, task: Task) -> bool:
        if task.create_at <= 0:
            return False
        now = time.time()
        if task.is_train:
            # 1天内的任务。
            return now - task.create_at > 24 * 3600

        return now - task.create_at > self.timeout

    def run(self) -> None:
        self._get_task()

    def start(self) -> None:
        super(TaskExecutor, self).start()
        self.exec_task()
