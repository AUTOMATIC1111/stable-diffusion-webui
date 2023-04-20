#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task_recv.py
# @Software: Hifive
import json
import random
import time
import typing
import redis_lock
from loguru import logger
from .task import Task
from tools.redis import RedisPool
from tools.model_hist import CkptLoadRecorder

try:
    from collections.abc import Iterable  # >=py3.10
except:
    from collections import Iterable  # <=py3.9

TaskQueuePrefix = "task_"
UpscaleCoeff = 100 * 1000
TaskScoreRange = (0, 100*UpscaleCoeff)
TaskTimeout = 24 * 3600


class TaskReceiver:

    def __init__(self, recoder: CkptLoadRecorder):
        self.model_recoder = recoder
        self.redis_pool = RedisPool()

    def _loaded_models(self):
        if self.model_recoder:
            return self.model_recoder.history()

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

    def search_task_with_id(self, rds, task_id) -> typing.Any:
        # redis > 6.2.0
        # meta = rds.getdel(task_id)
        meta = rds.get(task_id)
        if meta:
            rds.delete(task_id)
            return Task.from_json_str(meta)

    def _extract_queue_task(self, queue_name: str, retry: int = 1):
        queue_name = queue_name.decode('utf8') if isinstance(queue_name, bytes) else queue_name
        rds = self.redis_pool.get_connection()

        with redis_lock.Lock(rds, "task-lock-" + queue_name, expire=60) as locker:
            for _ in range(retry):
                now = int(time.time() * 1000)
                # min 最小为当前时间（ms）- VIP最大等级*放大系数（VIP提前执行权重）- 任务过期时间（1天）
                # max 为当前时间（ms） + 偏移量1秒
                min, max = now - TaskScoreRange[-1] - TaskTimeout*1000, now + 1000
                values = rds.zrangebyscore(
                    queue_name, min, max, start=0, num=1)
                task = None
                if values:
                    if isinstance(values, Iterable):
                        for v in values:
                            task = self.search_task_with_id(rds, v)
                            if task:
                                rds.zrem(queue_name, v)
                                break
                    else:
                        task = self.search_task_with_id(rds, v)
                        if task:
                            rds.zrem(queue_name, values)
                if task:
                    return task
                else:
                    rand = random.randint(0, 5) * 0.1
                    time.sleep(rand)

    def _get_queue_task(self, *model_hash: str):
        for sha256 in model_hash:
            queue_name = TaskQueuePrefix + sha256
            task = self._extract_queue_task(queue_name, 3)
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
            try:
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
            except:
                time.sleep(1)
                logger.exception("get task err")
