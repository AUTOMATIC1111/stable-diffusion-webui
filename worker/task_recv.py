#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task_recv.py
# @Software: Hifive
import os
import random
import shutil
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
OtherTaskQueueToken = 'others-'
TrainTaskQueueToken = 'train'
UpscaleCoeff = 100 * 1000
TaskScoreRange = (0, 100 * UpscaleCoeff)
TaskTimeout = 20 * 3600
Tmp = 'tmp'


def find_files_from_dir(directory, *args):
    extensions_ = [ex.lstrip('.').upper() for ex in args]
    for fn in os.listdir(directory):
        full_path = os.path.join(directory, fn)
        if os.path.isfile(full_path):
            if extensions_:
                _, ex = os.path.splitext(full_path)
                if ex.lstrip('.').upper() in extensions_:
                    yield full_path
            else:
                yield full_path
        elif os.path.isdir(full_path):
            for f in find_files_from_dir(full_path, *extensions_):
                yield f
            yield full_path


def clean_tmp(expired_days=1):
    if os.path.isdir(Tmp):
        now = time.time()
        files = [x for x in os.listdir(Tmp)]
        for fn in files:
            if not os.path.isfile(fn):
                continue
            mtime = os.path.getmtime(fn)
            if now > mtime + expired_days * 24 * 3600:
                try:
                    if os.path.isdir(fn):
                        shutil.rmtree(fn)
                    else:
                        os.remove(fn)
                except Exception:
                    logger.exception('cannot remove file!!')


class TaskReceiver:

    def __init__(self, recoder: CkptLoadRecorder, train_only: bool = False):
        self.model_recoder = recoder
        self.redis_pool = RedisPool()
        self.clean_tmp_time = time.time()
        self.train_only = train_only

    def _clean_tmp_files(self):
        now = time.time()
        if now - self.clean_tmp_time > 3600:
            self.clean_tmp_time = now
            clean_tmp()

    def _loaded_models(self):
        if self.model_recoder:
            return self.model_recoder.history()

    def _search_history_ckpt_task(self):
        if not self.train_only:
            model_hash_list = self._loaded_models()
            if model_hash_list:
                return self._get_queue_task(*model_hash_list)

    def _search_other_ckpt(self):
        if not self.train_only:
            model_hash_list = self._loaded_models()

            def sort_keys(x):
                x = str(x) if not isinstance(x, bytes) else x.decode('utf8')
                if x in model_hash_list:
                    return -1
                elif OtherTaskQueueToken in x:
                    return 0
                return 1

            keys = self._search_queue_names()
            sorted_keys = sorted(keys, key=sort_keys)
            for queue_name in sorted_keys:
                if TrainTaskQueueToken in queue_name:
                    continue

                task = self._extract_queue_task(queue_name)
                if task:
                    return task

    def _search_train_task(self):
        if self.train_only:
            keys = self._search_queue_names()
            for queue_name in keys:
                if TrainTaskQueueToken in queue_name:
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
        locker = redis_lock.Lock(rds, "task-lock-" + queue_name, expire=30)
        try:
            locker.acquire(blocking=True, timeout=40)
            for _ in range(retry):
                now = int(time.time() * 1000)
                # min 最小为当前时间（ms）- VIP最大等级*放大系数（VIP提前执行权重）- 任务过期时间（1天）
                # max 为当前时间（ms） + 偏移量1秒
                min, max = -1, now + 1000
                values = rds.zrangebyscore(
                    queue_name, min, max, start=0, num=1)
                task = None
                if values:
                    for v in values:
                        task = self.search_task_with_id(rds, v)
                        if task:
                            break
                    rds.zrem(queue_name, *values)
                if task:
                    return task
                elif not values:
                    rand = random.randint(0, 10) * 1
                    time.sleep(rand)
        except redis_lock.NotAcquired:
            pass
        except redis_lock.TimeoutTooLarge:
            pass
        except Exception as err:
            logger.exception("cannot get task from redis")
        finally:
            locker.release()

    def _get_queue_task(self, *model_hash: str):
        for sha256 in model_hash:
            queue_name = TaskQueuePrefix + sha256
            task = self._extract_queue_task(queue_name, 3)
            if task:
                return task

    def _search_queue_names(self):
        rds = self.redis_pool.get_connection()
        keys = rds.keys(TaskQueuePrefix + '*')
        return [k.decode('utf8') if isinstance(k, bytes) else k for k in keys]

    def _search_norm_task(self):
        t = self._search_history_ckpt_task()
        if t:
            return t
        t = self._search_other_ckpt()
        if t:
            return t

    def get_one_task(self, block: bool = True, sleep_time: float = 4) -> typing.Optional[Task]:
        while 1:
            st = time.time()
            if self.train_only:
                task = self._search_train_task()
            else:
                task = self._search_norm_task()
            if task:
                return task
            if not block:
                return None
            wait = sleep_time - time.time() + st
            if wait > 0:
                time.sleep(wait)

    def task_iter(self, sleep_time: float = 4) -> typing.Iterable[Task]:
        while 1:
            try:
                st = time.time()
                if self.train_only:
                    task = self._search_train_task()
                else:
                    task = self._search_norm_task()
                if task:
                    yield task
                wait = sleep_time - time.time() + st
                if wait > 0:
                    self._clean_tmp_files()
                    time.sleep(wait)
            except:
                time.sleep(1)
                logger.exception("get task err")
