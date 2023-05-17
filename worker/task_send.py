#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/3 9:54 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task_send.py
# @Software: Hifive
import os.path
import time

from tools.redis import RedisPool
from .task import Task
from .task_recv import TaskQueuePrefix
from .vip import VipLevel


class RedisSender:

    def __init__(self):
        self.redis_pool = RedisPool()

    def push_task(self, level: VipLevel, *tasks: Task):
        redis = self.redis_pool.get_connection()
        for task in tasks:
            if not task.model_hash:
                name, _ = os.path.splitext(os.path.basename(task.sd_model_path))
            else:
                name = task.model_hash[0:10]
            #  task_ + shorthash(前10位的sha256)
            queue = TaskQueuePrefix + name
            now = int(time.time() * 1000)
            meta = task.json()
            redis.set(task.id, meta, 3600*24*1)
            redis.zadd(queue, {
                task.id: int(level) * -100000 + now
            })

    def notify_train_task(self, task: Task):
        queue = 'checkpoint:train'
        rds = self.redis_pool.get_connection()
        rds.xadd(queue, {
            'task_id': task.id
        })
