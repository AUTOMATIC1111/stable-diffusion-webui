#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 5:25 PM
# @Author  : wangdongming
# @Site    :
# @File    : dumper.py
# @Software: Hifive
import abc
import json
import queue
import time
import typing
from loguru import logger
from threading import Thread
from tools.mgo import MongoClient
from tools.host import get_host_ip
from tools.redis import RedisPool
from worker.task import TaskProgress


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
        if '$set' in self.set:
            set_value = self.set['$set']
            if isinstance(set_value, dict) and 'task_progress' in set_value:
                logger.info(f"dumper task:{self.id} and progress {set_value['task_progress']}.")

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
        self.queue = queue.Queue(maxsize=100)
        self._stop_flag = False
        self._dump_now = False
        self._last_dump_time = 0
        self.redis_pool = RedisPool()

    def _set_cache(self, info: DumpInfo):
        try:
            rds = self.redis_pool.get_connection()
            rds.set(info.id.replace("task_id=", ""), json.dumps(info.set['$set']), 600)
        except Exception as err:
            logger.exception('cannot write to redis')

    def _get_queue_all(self):
        infos = {}
        while not self.queue.empty():
            info = self.queue.get()
            id = info.id
            if info.id not in infos:
                infos[id] = info
            elif isinstance(info, DumpInfo) and isinstance(infos[id], DumpInfo)\
                    and info.create_time > infos[id].create_time:
                infos[id] = info
            else:
                # 先进先出
                infos[id] = info

        return infos

    def run(self) -> None:
        while not self._stop_flag:
            now = time.time()
            if self._dump_now or now - self._last_dump_time > self.send_delay:
                self._last_dump_time = now
                if not self.queue.empty():
                    array = self._get_queue_all()
                    for info in array.values():
                        info.update_db(self.db)
                self._dump_now = False
            time.sleep(1)
            self.do_others()

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
        self._set_cache(info)
        if not self._dump_now:
            # 如果dump_now 已经是True,就不要覆盖
            # self._dump_now = task_progress.status >= TaskStatus.Prepare \
            #                  or task_progress.completed
            self._dump_now = task_progress.completed

    @abc.abstractmethod
    def progress_to_info(self, task_progress: TaskProgress) -> DumpInfo:
        raise NotImplementedError

    def do_others(self):
        pass

    def stop(self):
        self._stop_flag = True


class MongoTaskDumper(TaskDumper):

    def __init__(self, *args, **db_settings):
        mgo = MongoClient(**db_settings or {})
        mgo.collect.create_index('task_id', unique=True)
        mgo.collect.create_index('status')
        mgo.collect.create_index('task.user_id')
        self.clean_time = 0
        super(MongoTaskDumper, self).__init__(mgo)

    def progress_to_info(self, task_progress: TaskProgress) -> DumpInfo:
        v = task_progress.to_dict()
        v.update(
            {
                "task_id": task_progress.task.id,
                "ip": self.ip,
            }
        )

        info = DumpInfo(
            {"task_id": v['task_id']},
            {'$set': v},
            multi=False
        )
        return info

    def do_others(self):
        self.clean_timeout()

    def clean_timeout(self):
        now = int(time.time())
        if now - self.clean_time > 3600:
            tasks = self.db.collect.find({
                'status': 0,
                'task.create_at': {'$lt': now - 3600*12},
            })
            tasks = list(tasks)
            for task in tasks:
                task['status'] = -1
                task['task_desc'] = 'task timeout(auto clean).'
                self.db.update(
                    {"task_id": task['task_id']},
                    {'$set': task},
                    multi=False
                )

            self.clean_time = now


dumper = MongoTaskDumper()
dumper.start()



