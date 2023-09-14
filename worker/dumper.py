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
from datetime import datetime
from tools.mgo import MongoClient
from tools.host import get_host_ip
from tools.redis import RedisPool
from worker.task import TaskProgress
from tools.environment import pod_host, mongo_doc_expire_seconds


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
        self.ip = pod_host() or get_host_ip()
        self.send_delay = 10
        self.queue = queue.Queue(maxsize=100)
        self._stop_flag = False
        self._dump_now = False
        self._last_dump_time = 0
        self.redis_pool = RedisPool()

    def _set_cache(self, info: DumpInfo):
        try:
            rds = self.redis_pool.get_connection()
            data = dict(((k, v) for (k, v) in info.set['$set'].items() if not isinstance(v, datetime)))
            rds.set(info.id.replace("task_id=", ""), json.dumps(data), 1200)
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
        counter = 0
        while not self._stop_flag:
            try:
                now = time.time()
                if counter % 60 == 0:
                    logger.info("dumper waiting db record.")
                if counter > 120:
                    counter = 0
                if self._dump_now or now - self._last_dump_time > self.send_delay:
                    self._last_dump_time = now
                    if not self.queue.empty():
                        array = self._get_queue_all()
                        for info in array.values():
                            info.update_db(self.db)
                    self._dump_now = False
                    counter = 0
                time.sleep(1)
                self.do_others()
            except:
                logger.exception("unhandle err at dumper")
            finally:
                counter += 1

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
        self.before_push_info(task_progress, info)
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

    def before_push_info(self, task_progress, info):
        pass

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
        mgo.collect.create_index('task.create_at')
        mgo.collect.create_index('task.minor_type')
        mgo.collect.create_index('task.task_type')
        mgo.collect.create_index('task.model_name')
        mgo.collect.create_index('task.model_hash')

        image_cols = mgo.db['images']
        image_cols.create_index('task_id')
        image_cols.create_index('user_id')
        image_cols.create_index('create_at')
        image_cols.create_index('minor_type')
        image_cols.create_index('task_type')
        image_cols.create_index('model_name')
        image_cols.create_index('model_hash')
        image_cols.create_index('image_type')
        image_cols.create_index('group_id')
        doc_exp = mongo_doc_expire_seconds()
        if doc_exp > 0:
            logger.warning(f"set mongo doc expire after {doc_exp} seconds!")
            image_cols.create_index([("update_at", 1), ('expireAfterSeconds', doc_exp//2)])
            mgo.collect.create_index([("update_at", 1), ('expireAfterSeconds', doc_exp)])

        self.clean_time = 0
        super(MongoTaskDumper, self).__init__(mgo)

    def progress_to_info(self, task_progress: TaskProgress) -> DumpInfo:
        v = task_progress.to_dict()
        v.update(
            {
                "task_id": task_progress.task.id,
                "ip": self.ip,
                "update_at": datetime.now()
            }
        )

        info = DumpInfo(
            {"task_id": v['task_id']},
            {'$set': v},
            multi=False
        )
        return info

    def before_push_info(self, task_progress, info):
        self.write_images(task_progress)

    def do_others(self):
        self.clean_timeout()

    def clean_timeout(self):
        now = int(time.time())
        if now - self.clean_time > 3600:
            tasks = self.db.collect.find({
                'status': 0,
                'task.create_at': {'$lt': now - 3600*18},
            })
            tasks = list(tasks)
            for task in tasks:
                task['status'] = -1
                task['update_at'] = datetime.now()
                task['task_desc'] = 'task timeout(auto clean).'
                self.db.update(
                    {"task_id": task['task_id']},
                    {'$set': task},
                    multi=False
                )

            self.clean_time = now

    def write_images(self, task_progress: TaskProgress):
        if task_progress.completed and task_progress.result:
            r = task_progress.result
            flatten_images = []
            index = 0
            if 'grids' in r:
                for i, sample in enumerate(r['grids']['low']):
                    t = {'task_id': task_progress.task.id, 'model_hash': task_progress.task['model_hash'],
                         'user_id': task_progress.task.user_id, 'create_at': task_progress.task['create_at'],
                         'task_type': task_progress.task.task_type, 'minor_type': task_progress.task.minor_type,
                         'group_id': "", 'index': index, 'low_image': sample, 'image_type': 'grid',
                         'update_at': datetime.now(),
                         'high_image': r['grids']['high'][i]}
                    flatten_images.append(t)
                    index += 1
            if 'samples' in r:
                for i, sample in enumerate(r['samples']['low']):
                    t = {'task_id': task_progress.task.id, 'model_hash': task_progress.task['model_hash'],
                         'user_id': task_progress.task.user_id, 'create_at': task_progress.task['create_at'],
                         'task_type': task_progress.task.task_type, 'minor_type': task_progress.task.minor_type,
                         'group_id': "", 'index': index, 'low_image': sample, 'image_type': 'sample',
                         'high_image': r['samples']['high'][i],
                         'seed': task_progress.task['all_seed'][i],
                         'sub_seed': task_progress.task['all_sub_seed'][i],
                         'update_at': datetime.now(),
                         }
                    flatten_images.append(t)
                    index += 1
            if 'upscaler' in r:
                for i, sample in enumerate(r['all']['low']):
                    t = {'task_id': task_progress.task.id, 'model_hash': task_progress.task['model_hash'],
                         'user_id': task_progress.task.user_id, 'create_at': task_progress.task['create_at'],
                         'task_type': task_progress.task.task_type, 'minor_type': task_progress.task.minor_type,
                         'group_id': "", 'index': index, 'low_image': sample, 'image_type': 'sample',
                         'high_image': r['all']['high'][i],
                         'update_at': datetime.now(),
                         }
                    flatten_images.append(t)
                    index += 1

            if flatten_images:
                self.db.db['images'].insert_many(flatten_images)


dumper = MongoTaskDumper()
dumper.start()



