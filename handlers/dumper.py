#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 5:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : dumper.py
# @Software: Hifive
import abc
import queue
import time
import typing
from loguru import logger
from threading import Thread
from tools.mgo import MongoClient
from handlers.utils import get_host_ip
from worker.task import TaskProgress, TaskStatus


class DumpInfo:

    def __init__(self, query: typing.Union[str, typing.Mapping[str, str]], set: typing.Mapping, **kwargs):
        self.query = query
        self.set = set
        self.kwargs = kwargs
        self.create_time = time.time_ns()

    @property
    def id(self):
        if isinstance(self.query, str):
            return self.query
        else:
            sorted_items = sorted(self.query.items(), key=lambda x: x[0])
            return '&'.join((f"{k}={v}" for k, v in sorted_items))

    def update_db(self, db):
        logger.info(f"dumper task:{self.id}.")
        db.update(
            self.query,
            self.set,
            **self.kwargs
        )


class TaskDumper(Thread):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if not cls.__singleton:
            cls.__singleton = super(TaskDumper, cls).__new__(cls, *args, **kwargs)
        return cls.__singleton

    def __init__(self, db):
        if not hasattr(db, 'update'):
            raise TypeError('db not found update function')

        super(TaskDumper, self).__init__(name='task-dumper')
        self.db = db
        self.ip = get_host_ip()
        self.send_delay = 10
        self._last_dump_time = 0
        self.queue = queue.Queue(maxsize=100)
        self._stop = False
        self._dump_now = False

    def _get_queue_all(self):
        infos = {}
        while not self.queue.empty():
            info = self.queue.get()
            id = info.id
            if info.id not in infos:
                infos[id] = info
            elif info.create_time > infos[id].create_time:
                infos[id] = info
        return infos

    def run(self) -> None:
        while not self._stop:
            now = time.time()
            if self._dump_now or self._last_dump_time + self.send_delay > now:
                self._last_dump_time = now
                if not self.queue.empty():
                    array = self._get_queue_all()
                    for info in array.values():
                        info.update_db(self.db)
                self._dump_now = False
            time.sleep(1)

    def dump_task_progress(self, task_progress: TaskProgress):
        # try:
        #     v = task_progress.to_dict()
        #     v.update(
        #         {
        #             "task_id": task_progress.task.id,
        #             "ip": self.ip,
        #             "latency": int(time.time()) - task_progress.task.create_at,
        #         }
        #     )
        #     self.mgo.update(
        #         {"task_id": v['task_id']},
        #         {'$set': v},
        #         multi=False
        #     )
        #     return 1
        # except Exception:
        #     loguru.logger.exception('dump task failed.')
        #     return 0

        info = self.progress_to_info(task_progress)
        self.queue.put(info)
        if not self._dump_now:
            # 如果dump_now 已经是True,就不要覆盖
            self._dump_now = task_progress.task_progress > 0

    @abc.abstractmethod
    def progress_to_info(self, task_progress: TaskProgress) -> DumpInfo:
        raise NotImplementedError

    def stop(self):
        self._stop = True


class MongoTaskDumper(TaskDumper):

    def __init__(self):
        mgo = MongoClient()
        mgo.collect.create_index('task_id', unique=True)
        super(MongoTaskDumper, self).__init__(mgo)

    def progress_to_info(self, task_progress: TaskProgress) -> DumpInfo:
        v = task_progress.to_dict()
        v.update(
            {
                "task_id": task_progress.task.id,
                "ip": self.ip,
                "latency": int(time.time()) - task_progress.task.create_at,
            }
        )

        info = DumpInfo(
            {"task_id": v['task_id']},
            {'$set': v},
            multi=False
        )
        return info


dumper = MongoTaskDumper()
dumper.start()


