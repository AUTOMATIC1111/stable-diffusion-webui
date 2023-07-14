#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 3:18 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task_recv.py
# @Software: Hifive
import enum
import hashlib
import json
import os
import random
import shutil
import time
import typing
import uuid
import redis_lock
import requests
from loguru import logger
from .task import Task
from datetime import datetime, timedelta
from modules.shared import cmd_opts
from tools.redis import RedisPool
from collections import defaultdict
from tools.model_hist import CkptLoadRecorder
from tools.gpu import GpuInfo
from tools.wrapper import timed_lru_cache
from tools.host import get_host_name
from modules.shared import mem_mon as vram_mon
from apscheduler.schedulers.background import BackgroundScheduler
from tools.environment import get_run_train_time_cfg, get_worker_group, get_gss_count_api,\
    Env_Run_Train_Time_Start, Env_Run_Train_Time_End, is_flexible_worker, get_worker_state_dump_path

try:
    from collections.abc import Iterable  # >=py3.10
except:
    from collections import Iterable  # <=py3.9

TaskQueuePrefix = "task_"
OtherTaskQueueToken = TaskQueuePrefix + 'others'
TrainTaskQueueToken = 'train'
UpscaleCoeff = 100 * 1000
TaskScoreRange = (0, 100 * UpscaleCoeff)
TaskTimeout = 20 * 3600 if not cmd_opts.train_only else 48 * 3600
Tmp = 'tmp'
SDWorkerZset = 'sd-workers'
ElasticResWorkerFlag = "[ElasticRes]"


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


class TaskReceiverState(enum.Enum):
    Running = "running"
    Idle = "idle"


class TaskReceiverRecorder:

    def __init__(self, file_path: str = None, interval=60):
        self.file_path = file_path or get_worker_state_dump_path("worker_state.log")
        self.timestamp = int(time.time())
        self.current = TaskReceiverState.Running
        self.interval = interval
        self.dump_ts = 0

    def set_state(self, state: TaskReceiverState):
        if state != self.current:
            self.current = state
            self.timestamp = int(time.time())

    def write(self):
        if int(time.time()) - self.dump_ts > self.interval:
            # 记录时间大于状态修改时间说明该状态已经记录
            if self.dump_ts < self.timestamp:
                with open(self.file_path, "w+") as f:
                    f.write(json.dumps({
                        "status": str(self.current.value),
                        "timestamp": self.timestamp
                    }))
            self.dump_ts = int(time.time())


def register_worker(worker_id):
    try:
        pool = RedisPool()
        conn = pool.get_connection()
        conn.setex(worker_id, 300, 60)
        conn.zadd(SDWorkerZset, {
            worker_id: int(time.time())
        })
        conn.expire(SDWorkerZset, timedelta(hours=1))
        pool.close()
    except:
        pass


class TaskReceiver:

    def __init__(self, recoder: CkptLoadRecorder, train_only: bool = False):
        self.release_flag = None
        self.model_recoder = recoder
        self.redis_pool = RedisPool()
        self.clean_tmp_time = time.time()
        self.train_only = train_only
        # self.worker_id = self._worker_id()
        self.is_elastic = is_flexible_worker()
        self.recorder = TaskReceiverRecorder()
        self.worker_id = self._worker_id()

        run_train_time_cfg = get_run_train_time_cfg()
        run_train_time_start = run_train_time_cfg[Env_Run_Train_Time_Start]
        run_train_time_end = run_train_time_cfg[Env_Run_Train_Time_End]

        def formate_day_of_time(day_of_time: int):
            if day_of_time < 0:
                return 24 + day_of_time
            return day_of_time

        run_train_time_start = formate_day_of_time(int(run_train_time_start) - 8 if run_train_time_start else 15)
        run_train_time_end = formate_day_of_time(int(run_train_time_end) - 8 if run_train_time_end else 23)

        self.run_train_time_start = min(run_train_time_start, run_train_time_end)
        self.run_train_time_end = max(run_train_time_start, run_train_time_end)

        logger.info(f"worker id:{self.worker_id}, train work receive clock:{self.run_train_time_start} - {self.run_train_time_end}")

        self.register_time = 0
        self.local_cache = {}
        self.timer = BackgroundScheduler()

        self.timer.add_job(register_worker, 'interval', seconds=30, args=[self.worker_id])
        self.timer.start()

    def _worker_id(self):
        group_id = get_worker_group()
        nvidia_video_card_id = '&'.join(GpuInfo().names)

        # int(str(uuid.uuid1())[-4:], 16)
        hostname = get_host_name()
        # hostname = 'sdplus-saas-qa-568ff9745c-rcwm6'
        try:
            int(hostname[-16:-6], 16)
            hostname = 'Host:' + hostname
        except:
            hostname = None

        print(f"hostname:{hostname}, vedio id:{nvidia_video_card_id}")
        if not nvidia_video_card_id:
            nvidia_video_card_id = hostname or uuid.uuid1()
        else:
            index = nvidia_video_card_id.index("GPU")

            if not hostname:
                hash_md5 = hashlib.md5()
                hash_md5.update(nvidia_video_card_id.encode())
                nvidia_video_card_id = nvidia_video_card_id[:index] + hash_md5.hexdigest()[:8] + "-" + uuid.uuid4().hex
            else:
                nvidia_video_card_id = nvidia_video_card_id[:index] + "-" + hostname
        nvidia_video_card_id = nvidia_video_card_id.replace("(", "-").replace(")", "")

        if is_flexible_worker():
            nvidia_video_card_id = ElasticResWorkerFlag + nvidia_video_card_id

        return f"{group_id}:{nvidia_video_card_id}"

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
        # 弹性不训练
        if self.is_elastic:
            return

        if self.train_only or self._can_gener_img_worker_run_train():
            keys = self._search_queue_names()
            for queue_name in keys:
                if TrainTaskQueueToken in queue_name:
                    task = self._extract_queue_task(queue_name)
                    if task:
                        return task

    def _can_gener_img_worker_run_train(self):
        # 默认23点~凌晨5点(UTC 15~21)可以运行TRAIN
        utc = datetime.utcnow()

        if self.run_train_time_start <= utc.hour < self.run_train_time_end:
            logger.info(f"worker receive train task")

            group_workers = self.get_group_workers()
            group_id = get_worker_group()
            workers = group_workers.get(group_id) or []
            if workers:
                # 1/5的WOEKER 生图，剩下的执行训练。
                run_train_worker_num = len(workers) // 5
                if run_train_worker_num >= 1:
                    run_train_workers = workers[run_train_worker_num:]
                else:
                    run_train_workers = []

                logger.info(f"run train task worker ids:{';'.join(run_train_workers)}, current id:{self.worker_id}")
                run_train_worker_flag = self.worker_id in run_train_workers
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] GPU free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                if run_train_worker_flag and free / 2 ** 30 > 16:
                    logger.info(">>> worker can run train task.")

                return run_train_worker_flag

        return False

    def search_task_with_id(self, rds, task_id) -> typing.Any:
        # redis > 6.2.0
        # meta = rds.getdel(task_id)
        meta = rds.get(task_id)
        if meta:
            task_id = task_id.decode('utf8') if isinstance(task_id, bytes) else task_id
            rds.setex(f"task:worker:{task_id}", timedelta(hours=2), self.worker_id)
            return Task.from_json_str(meta)

    def _extract_queue_task(self, queue_name: str, retry: int = 1):
        queue_name = queue_name.decode('utf8') if isinstance(queue_name, bytes) else queue_name
        rds = self.redis_pool.get_connection()

        locker = redis_lock.Lock(rds, "task-lock-" + queue_name, expire=10)
        locked = False
        try:
            locker.acquire(blocking=True, timeout=3)
            locked = True
            for _ in range(retry):
                now = int(time.time() * 1000)
                # min 最小为当前时间（ms）- VIP最大等级*放大系数（VIP提前执行权重）- 任务过期时间（1天）
                # max 为当前时间（ms） + 偏移量1秒
                min, max = -1, now + 1000
                values = rds.zrangebyscore(
                    queue_name, min, max, start=0, num=1)
                task = None
                if values:
                    rds.zrem(queue_name, *values)
                    for v in values:
                        task = self.search_task_with_id(rds, v)
                        if task:
                            break
                if task:
                    return task
                elif not values:
                    rand = random.randint(0, 10) * 0.1 * retry
                    if (rand < 0.5):
                        continue
                    time.sleep(rand)
        except redis_lock.NotAcquired:
            locked = False
        except redis_lock.TimeoutTooLarge:
            locked = False
        except Exception as err:
            logger.exception("cannot get task from redis")
        finally:
            try:
                if locked:
                    locker.release()
            except:
                pass

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

    def _search_task(self):
        t = self._search_history_ckpt_task()
        if t:
            return t
        t = self._search_other_ckpt()
        if t:
            return t
        if self._can_gener_img_worker_run_train():
            t = self._search_train_task()
            if t:
                return t

    def get_one_task(self, block: bool = True, sleep_time: float = 4) -> typing.Optional[Task]:
        while 1:
            st = time.time()
            if self.train_only:
                task = self._search_train_task()
            else:
                task = self._search_task()
            if task:
                return task
            if not block:
                return None
            wait = sleep_time - time.time() + st
            if wait > 0:
                time.sleep(wait)
            self.register_worker()

    def task_iter(self, sleep_time: float = 2) -> typing.Iterable[Task]:
        while 1:
            try:
                st = time.time()

                # 释放弹性资源，不再获取任务主动
                if self.release_elastic_res_state():
                    logger.info("release elastic resource...")
                    task = None
                    time.sleep(1)
                else:
                    if self.train_only:
                        task = self._search_train_task()
                    else:
                        task = self._search_task()

                if task:
                    self.recorder.set_state(TaskReceiverState.Running)
                    yield task
                else:
                    self.recorder.set_state(TaskReceiverState.Idle)

                wait = sleep_time - time.time() + st
                if wait > 0:
                    self._clean_tmp_files()
                    time.sleep(wait)

            except:
                time.sleep(1)
                logger.exception("get task err")
            finally:
                self.register_worker()
                self.write_worker_state()

    def register_worker(self):
        if time.time() - self.register_time > 120:
            try:
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] GPU free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
                # conn = self.redis_pool.get_connection()
                # conn.setex(self.worker_id, 300, 60)
                # conn.zadd(SDWorkerZset, {
                #     self.worker_id: int(time.time())
                # })
                # conn.expire(SDWorkerZset, timedelta(hours=1))
                self.register_time = time.time()
            except:
                return False
        return True

    @timed_lru_cache(60)
    def get_all_workers(self):
        now = int(time.time())
        rds = self.redis_pool.get_connection()
        keys = rds.zrangebyscore(SDWorkerZset, now - 300, now)
        worker_ids = [k.decode('utf8') if isinstance(k, bytes) else k for k in keys]

        def get_work_id_num(x: str):
            # def hashs(s):
            #     hash_md5 = hashlib.md5()
            #     hash_md5.update(s.encode())
            #
            #     return hash_md5.hexdigest()[:8]
            if 'Host:' in x:
                idx = x.index('Host:')
                return x[idx:]

            return x.replace("-", "")[-8:]

        worker_ids = sorted(worker_ids, key=get_work_id_num)
        return worker_ids

    def get_group_workers(self):
        workers = self.get_all_workers()
        gw = defaultdict(list)
        for w in workers:
            array = w.split(':')
            key = array[0]
            gw[key].append(":".join(array[1:]))
        return gw

    def is_elastic_worker(self, worker_id: str) -> bool:
        return ElasticResWorkerFlag in worker_id

    def write_worker_state(self):
        self.recorder.write()

    def release_elastic_res_state(self):
        if self.is_elastic:
            if time.time() - self.register_time > 120:
                url = get_gss_count_api()
                if url:
                    try:
                        resp = requests.get(url, timeout=5)
                        json_data = resp.json() or {}
                        data = json_data.get('data') or {}
                        need_workers = data.get('need_workers', 0)
                        self.release_flag = need_workers == 0
                    except Exception as ex:
                        logger.exception('cannot get elastic workers')
                        self.release_flag = False
        else:
            self.release_flag = False
        return self.release_flag

    def close(self):
        self.timer.shutdown()
