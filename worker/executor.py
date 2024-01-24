#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 2:17 PM
# @Author  : wangdongming
# @Site    : 
# @File    : executor.py
# @Software: Hifive
import os.path
import queue
import random
import time
import typing
from queue import Queue
from loguru import logger
from tools import safety_clean_tmp
from tools.disk import tidy_model_caches
from worker.task import Task, TaskProgress
from modules.shared import mem_mon as vram_mon, models_path
from worker.handler import TaskHandler
from modules.devices import torch_gc
from worker.task_recv import TaskReceiver, TaskTimeout
from threading import Thread, Condition, Lock
from tools.model_hist import CkptLoadRecorder
from tools.environment import Env_DontCleanModels
from worker.k8s_health import write_healthy, system_exit, process_health


class TaskExecutor(Thread):

    def __init__(self, ckpt_recorder: CkptLoadRecorder, timeout=0, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None, train_only=False):
        self.recorder = ckpt_recorder
        self._handlers = {}
        self.receiver = TaskReceiver(ckpt_recorder, train_only, self.can_exec_task, self.before_receive_task)
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
        for h in self._handlers.values():
            h.close()
        self.receiver.close()

    def stop(self):
        self.__stop = True

    def can_exec_task(self, task: Task) -> bool:
        types = self._handlers.keys()
        return task.task_type in types

    def add_handler(self, *handlers: TaskHandler):
        for handler in handlers:
            if not handler.enable:
                logger.warning(f"handler disable:{handler.task_type.name}")
            self._handlers[handler.task_type] = handler

    def get_handler(self, task: Task) -> typing.Optional[TaskHandler]:
        return self._handlers.get(task.task_type)

    def error_handler(self, task: Task, ex: Exception):
        logger.error(f'exec task failed: {task.desc()}, ex: {ex}')

    def nofity(self):
        if getattr(self.not_busy, "value", 0) == 0:
            return

        with self.not_busy:
            self.not_busy.notify_all()
            logger.debug("notify receiver ")
            setattr(self.not_busy, "value", 0)

    def task_progress(self, p: TaskProgress):
        if p.pre_task_completed():
            self.nofity()

    def exec_task(self):
        write_healthy(True)
        handlers = [x.name for x in self._handlers.keys()]
        logger.info(f"executor start with:{','.join(handlers)}")
        while not self.__stop:
            try:
                logger.info(f"====>>> start receive and execute task")
                task = self.queue.get(timeout=10)
                if not task:
                    continue

                logger.info(f"====>>> receive task:{task. desc()}")
                logger.info(f"====>>> model history:{self.recorder.history()}")

                handler = self.get_handler(task)
                if not handler:
                    self.error_handler(task, Exception('can not found task handler'))
                # 判断TASK超时。
                if self._is_timeout(task):
                    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.create_at))
                    handler.set_failed(task, f'task time out(task create time:{create_time}, now:{now})')
                    self.nofity()
                    continue
                handler(task, progress_callback=self.task_progress)

            except queue.Empty:
                if random.randint(1, 10) < 3:
                    logger.info('task queue is empty...')
                if not self.is_alive():
                    logger.info('task receiver dead')
                    free, total = vram_mon.cuda_mem_get_info()
                    system_exit(free, total, coercive=True)
                continue
            except Exception:
                logger.exception("executor err")
                self.nofity()

        logger.info('executor stopping...')
        write_healthy(False)
        self._close()

    def before_receive_task(self, *task_ids):
        def has_train_task():
            for id in task_ids:
                if isinstance(id, bytes):
                    id = id.decode()
                if 'train' in id:
                    return True

        train_task = has_train_task()

        if train_task:
            logger.info("before receive train task, check memory info...")
            ok = torch_gc()
            free, total = vram_mon.cuda_mem_get_info()
            system_exit(free, total, threshold_b=12, coercive=not ok)
        elif random.randint(1, 10) < 3:
            logger.info("before receive task, check memory info...")
            ok = torch_gc()
            free, total = vram_mon.cuda_mem_get_info()
            system_exit(free, total, coercive=not ok)
            process_health()

        return True

    def _get_persist_model_hashes(self):
        rds = self.receiver.get_redis_cli()
        hashes = rds.lrange('persist_models', 0, -1)
        return hashes

    def _get_task(self):
        while self.is_alive() and not self.receiver.closed and not self.__stop:
            with self.not_busy:
                if not self.queue.full():
                    try:
                        for task in self.receiver.task_iter():
                            if random.randint(1, 10) < 3:
                                # 释放磁盘空间
                                safety_clean_tmp()
                                model_hashes = self._get_persist_model_hashes()
                                tidy_model_caches(models_path, persist_model_hashes=model_hashes)
                            logger.info(f"====>>> preload task:{task.id}")
                            self.queue.put(task)
                            logger.info(f"====>>> push task:{task.id}")
                            if isinstance(task, Task):
                                logger.debug(f"====>>> waiting task:{task.id}, stop receive.")
                                setattr(self.not_busy, "value", 1)
                                try:
                                    timeout = 3600 * 16
                                    self.not_busy.wait(timeout=timeout)
                                    logger.info(f"====>>> acquire locker, time out:{timeout} seconds")
                                except Exception as err:
                                    free, total = vram_mon.cuda_mem_get_info()
                                    logger.exception("executor cannot require locker, quit...")
                                    self.receiver.close()
                                    system_exit(free, total, True)
                                    break
                                self.receiver.decr_train_concurrency(task)
                                logger.debug(f"====>>> waiting task:{task.id}, begin receive.")
                    except Exception:
                        logger.exception("receive task failed, restart app...")
                        self.receiver.close()
                        self.__stop = True
                        system_exit(0, 0, True)
                else:
                    self.not_busy.wait()
        logger.info("=======> task receiver quit!!!!!!")
        self.__stop = True

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

