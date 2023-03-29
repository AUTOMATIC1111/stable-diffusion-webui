#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task_recv.py
# @Software: Hifive
import json
import time
import typing

from task import Task
from tools.redis import RedisPool
from collections import Iterable
from tools.model_hist import ModelHistory

TaskScoreRange = (0, 100)
TaskQueuePrefix = "task_"


class TaskReceiver:

    def __init__(self, history: ModelHistory):
        self.model_history = history
        self.redis_pool = RedisPool()

    def _loaded_models(self):
        if self.model_history:
            return self.model_history.history()

    def _search_history_ckpt_task(self):
        model_hash_list = self._loaded_models()
        if model_hash_list:
            return self._get_queue_task(*model_hash_list)

    def _search_other_ckpt(self):
        keys = self._search_queue_names()
        for queue_name in keys:
            task = self._extract_queue_task(queue_name)
            if task:
                return task

    def _extract_queue_task(self, queue_name: str):
        rds = self.redis_pool.get_connection()
        values = rds.zrevrangebyscore(
            queue_name, TaskScoreRange[-1], TaskScoreRange[0], num=1)
        task = None
        if values:
            if isinstance(values, Iterable):
                for v in values:
                    task = Task.from_json_str(v)
                    if task:
                        break
            else:
                task = Task.from_json_str(values)
        if task:
            return task

    def _get_queue_task(self, *model_hash: str):
        for sha256 in model_hash:
            queue_name = TaskQueuePrefix + sha256
            task = self._extract_queue_task(queue_name)
            if task:
                return task

    def _search_queue_names(self):
        rds = self.redis_pool.get_connection()
        keys = rds.keys(TaskQueuePrefix + '*')
        return keys

    def get_one_task(self, block: bool = True, sleep_time: float = 0.5) -> typing.Optional[Task]:
        while 1:
            st = time.time()
            t = self._search_history_ckpt_task()
            if t:
                return t
            t = self._search_other_ckpt()
            if t:
                return t
            if not block:
                return None
            wait = sleep_time - time.time() + st
            if wait > 0:
                time.sleep(wait)

    def task_iter(self, sleep_time: float = 0.5) -> typing.Iterable[Task]:
        while 1:
            st = time.time()
            t = self._search_history_ckpt_task()
            if t:
                yield t
            t = self._search_other_ckpt()
            if t:
                yield t
            wait = sleep_time - time.time() + st
            if wait > 0:
                time.sleep(wait)
