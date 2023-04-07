#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 5:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : dumper.py
# @Software: Hifive
import bson
import loguru
from tools.mgo import MongoClient
from .utils import get_host_ip
from worker.task import TaskProgress


class TaskDumper:
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls.__singleton:
            cls.__singleton = super(TaskDumper, cls).__new__(cls, *args, **kwargs)
        return cls.__singleton

    def __init__(self):
        self.mgo = MongoClient()
        self.mgo.collect.create_index('task_id', unique=True)
        self.ip = get_host_ip()

    def dump_task_progress(self, task_progress: TaskProgress):
        try:
            v = task_progress.to_dict()
            v.update(
                {
                    "task_id": task_progress.task.id,
                    "ip": self.ip
                }
            )
            self.mgo.update(
                {"task_id": v['task_id']},
                {'$set': v},
                multi=False
            )
            return 1
        except Exception:
            loguru.logger.exception('dump task failed.')
            return 0
