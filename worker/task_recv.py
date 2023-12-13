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
from tools.environment import get_run_train_time_cfg, get_worker_group, get_gss_count_api, run_train_ratio, \
    Env_Run_Train_Time_Start, Env_Run_Train_Time_End, is_flexible_worker, get_worker_state_dump_path, \
    is_task_group_queue_only, get_maintain_env

try:
    from collections.abc import Iterable  # >=py3.10
except:
    from collections import Iterable  # <=py3.9

TaskQueuePrefix = "task_"
TaskPreQueuePrefix = 'task-pre_'
OtherTaskQueueToken = TaskQueuePrefix + 'others'
TrainTaskQueueToken = 'train'
UpscaleCoeff = 100 * 1000
TaskScoreRange = (0, 100 * UpscaleCoeff)
TaskTimeout = 20 * 3600 if not cmd_opts.train_only else 48 * 3600
SDWorkerZset = 'sd-workers'
ElasticResWorkerFlag = "[ElasticRes]"
TrainOnlyWorkerFlag = "[TrainOnly]"
MaintainKey = get_maintain_env()
MaintainReadyKey = MaintainKey + "-ready"


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

    def idle_time(self):
        with open(self.file_path) as f:
            d = json.loads(f.readline())
            if d['status'] == TaskReceiverState.Idle.value:
                return int(time.time()) - d['timestamp']


def register_worker(worker_id: typing.Mapping):
    try:

        now = int(time.time())
        pool = RedisPool()
        conn = pool.get_connection()
        conn.zadd(SDWorkerZset, {
            worker_id: now
        })
        conn.expire(SDWorkerZset, timedelta(hours=1))
        conn.zremrangebyscore(SDWorkerZset, 0, now - 3600 * 24)
        pool.close()
    except:
        logger.exception("cannot register worker")


class TaskReceiver:

    def __init__(self, recoder: CkptLoadRecorder, train_only: bool = False,
                 task_received_callback: typing.Callable = None,
                 before_pop_task: typing.Callable = None):
        self.release_flag = None
        self.model_recoder = recoder
        self.redis_pool = RedisPool()
        self.clean_tmp_time = time.time()
        self.train_only = train_only
        # self.worker_id = self._worker_id()
        self.is_elastic = is_flexible_worker()
        self.recorder = TaskReceiverRecorder()
        self.group_id = get_worker_group()
        self.task_score_limit = 5 if cmd_opts.lowvram else (10 if cmd_opts.medvram else -1)
        self.task_received_callback = task_received_callback
        self.before_pop_task = before_pop_task

        run_train_time_cfg = get_run_train_time_cfg()
        run_train_time_start = run_train_time_cfg[Env_Run_Train_Time_Start]
        run_train_time_end = run_train_time_cfg[Env_Run_Train_Time_End]

        def formate_day_of_time(day_of_time: int):
            if day_of_time < 0:
                return 24 + day_of_time
            return day_of_time

        run_train_time_start = formate_day_of_time(int(run_train_time_start) - 8 if run_train_time_start else 10)
        run_train_time_end = formate_day_of_time(int(run_train_time_end) - 8 if run_train_time_end else 23)

        self.run_train_time_start = min(run_train_time_start, run_train_time_end)
        self.run_train_time_end = max(run_train_time_start, run_train_time_end)

        self.register_time = 0
        self.local_cache = {}
        self.timer = BackgroundScheduler()

        worker_info = self._worker_info()
        self.timer.add_job(register_worker, 'interval', seconds=30, args=[worker_info['worker_id']])
        self.exception_ts = 0
        self.closed = False
        self.is_task_group_queue_only = is_task_group_queue_only()
        self.worker_id = self._worker_id()
        self.timer.start()
        logger.info(
            f"worker id:{self.worker_id}, train work receive clock:"
            f"{self.run_train_time_start} - {self.run_train_time_end}")

    def _worker_id(self):
        info = self._worker_info()

        return info['worker_id']

    @timed_lru_cache(300)
    def _worker_info(self):
        group_id = get_worker_group()
        gpu_names = '&'.join(GpuInfo().names)
        nvidia_video_card_id = gpu_names

        # int(str(uuid.uuid1())[-4:], 16)
        hostname = get_host_name()
        # hostname = 'sdplus-saas-qa-568ff9745c-rcwm6'
        try:
            x = hostname.split("-")[-2]
            int(x, 16)
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
        if self.train_only:
            nvidia_video_card_id = TrainOnlyWorkerFlag + nvidia_video_card_id
            if is_flexible_worker():
                raise OSError('elastic resource cannot run with train only mode')
        worker_id = f"{group_id}:{self.task_score_limit}.{nvidia_video_card_id}"
        exec_train = self.train_only or self._can_gener_img_worker_run_train(worker_id)
        model_hash_list = self.model_recoder.history()

        return {
            'worker_id': worker_id,
            'flexible': is_flexible_worker(),
            'video_id': nvidia_video_card_id,
            'group': group_id,
            'max_task_score': self.task_score_limit,
            'resource': gpu_names.replace(' ', "-"),
            'exec_train_task': exec_train,
            'model_hash_list': model_hash_list,
        }

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

    def _search_group_task_queue(self):
        '''
        搜索指定资源的队列。
        '''
        info = self._worker_info()
        resource_name = info['resource']

        return self._get_queue_task(resource_name)

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

    def _can_gener_img_worker_run_train(self, worker_id=None):
        # 默认23点~凌晨5点(UTC 15~21)可以运行TRAIN
        utc = datetime.utcnow()

        if self.run_train_time_start <= utc.hour < self.run_train_time_end:
            logger.info(f"worker receive train task")

            group_workers = self.get_group_workers()
            group_id = get_worker_group()
            workers = group_workers.get(group_id) or []
            if workers:
                def can_exec_train(worker_id: str):
                    if TrainOnlyWorkerFlag in worker_id:
                        return False

                    start = worker_id.index(':') if ':' in worker_id else 0
                    end = worker_id.index(".")
                    if end > start > 0:
                        try:
                            limit = int(worker_id[start: end])
                            return limit == -1
                        except:
                            return True

                    return True

                # 1/5的WOEKER 生图，剩下的执行训练。
                workers = [w for w in workers if can_exec_train(w)]

                run_train_worker_num = int(len(workers) * run_train_ratio())

                if run_train_worker_num >= 1:
                    run_train_workers = workers[:run_train_worker_num]
                else:
                    run_train_workers = []
                if not worker_id:
                    no_group_worker_id = self.worker_id.replace(group_id + ":", '')
                else:
                    no_group_worker_id = worker_id.replace(group_id + ":", '')
                logger.info(f"run train task worker ids:{';'.join(run_train_workers)}, current id:{no_group_worker_id}")
                run_train_worker_flag = no_group_worker_id in run_train_workers
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] GPU free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                if run_train_worker_flag and free / 2 ** 30 > 16:
                    logger.info(f">>> worker receive train task, train worker count:{run_train_worker_num}")

                return run_train_worker_flag

        return False

    def search_task_with_id(self, rds, task_id, queue_name, task_score) -> typing.Any:
        # redis > 6.2.0
        # meta = rds.getdel(task_id)
        if self.task_score_limit > 0:
            arr = task_id.split('_')
            if len(arr) == 2:
                try:
                    score = int(arr[-1])
                    if self.task_score_limit < score:
                        print(f"====> task:{task_id} out of limit({self.task_score_limit}).")
                        return

                except:
                    pass

        meta = rds.get(task_id)
        if meta:
            t = Task.from_json_str(meta)
            if t.is_train:
                paralle_count = t.get('paralle_count', 0)
                if paralle_count > 0:
                    can_exec = self.can_exec_train_task(t.user_id, paralle_count)

                    if not can_exec:
                        delay = 0
                        logger.info(f"repush task {t.id} to {queue_name} and score:{task_score + delay}")
                        time.sleep(10)
                        self.repush_task(t.id, queue_name, task_score + delay)
                        time.sleep(2)
                        logger.debug(f"return none task")
                        return
                    else:
                        self.incr_train_concurrency(t.user_id)

            if callable(self.task_received_callback) and not self.task_received_callback(t):
                logger.info("receive callback repush task.")
                self.repush_task(t.id, queue_name, task_score)
                return

            task_id = task_id.decode('utf8') if isinstance(task_id, bytes) else task_id
            rds.setex(f"task:worker:{task_id}", timedelta(hours=2), self.worker_id)
            return t

    def _check_cluster_status(self):
        '''
        检测集群服务状态，如果是维护状态就陷入睡眠
        '''
        # 维护就绪时间
        current_timestamp = int(time.time())
        isFirst = True
        while 1:
            try:
                rds = self.redis_pool.get_connection()
                flag = rds.get(MaintainKey) or "0"
                awake_ts = int(flag)
                if time.time() < awake_ts:
                    time_array = time.localtime(awake_ts)
                    date_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
                    # 维护就绪状态写入reids
                    if isFirst:
                        rds.zadd(MaintainReadyKey, {self.worker_id: current_timestamp})
                        isFirst = False
                    logger.info(f"[Maintain] task receiver sleeping till: {date_time}...")
                    time.sleep(10)
                else:
                    # 如果是维护状态，并且维护结束，删除维护就绪
                    if not isFirst:
                        rds.zrem(MaintainReadyKey, self.worker_id)
                    break
            except Exception as err:
                logger.warning(f"cannot got cluster status:{err}")
                break

    def _extract_queue_task(self, queue_name: str, retry: int = 1):
        queue_name = queue_name.decode('utf8') if isinstance(queue_name, bytes) else queue_name
        rds = self.redis_pool.get_connection()

        locker = redis_lock.Lock(rds, "task-lock-" + queue_name, expire=10)
        locked = False
        try:
            logger.debug("===> acquire locker: task-lock-" + queue_name)
            locker.acquire(blocking=True, timeout=3)
            locked = True
            # self._preload_task(queue_name)
            for _ in range(retry):
                now = int(time.time() * 1000)
                # min 最小为当前时间（ms）- VIP最大等级*放大系数（VIP提前执行权重）- 任务过期时间（1天）
                # max 为当前时间（ms） + 偏移量1秒
                min, max = -1, now + 1000
                values = rds.zrangebyscore(
                    queue_name, min, max, start=0, num=1, withscores=True)
                task = None
                if values:
                    names = [v[0] for v in values]
                    # before receive task
                    if callable(self.before_pop_task):
                        r = self.before_pop_task(*names)
                        if not r:
                            continue

                    rds.zrem(queue_name, *names)
                    for name, score in values:

                        task = self.search_task_with_id(rds, name, queue_name, score)
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

    def _preload_task(self, queue_name: str):
        rds = self.redis_pool.get_connection()
        model_hash = queue_name[len(TaskQueuePrefix):]
        # 查询键：task-pre_{model_hash}*
        keys = rds.keys(TaskPreQueuePrefix + model_hash + '*')

        if not keys:
            return

        now = int(time.time() * 1000)
        min, max = -1, now + 1000
        # 当前任务队列长度
        count = rds.zcount(queue_name, min, max)
        for key in keys:
            concurrency = int(key.split('-')[-1])
            push_num = concurrency - count
            logger.info(f"[{queue_name}] preload task num:{push_num}.")
            # 从预加载队列LPOP task_id
            task_ids = rds.lpop(key, push_num)
            for task_id in task_ids:
                # 设置task str任务的meta信息的过期
                rds.expire(task_id, 3600 * 24)
                # 从pre队列的task id 导入到正式队列
                rds.zadd(queue_name, {
                    task_id: now
                })
                logger.info(f"preload task:{task_id} to {queue_name}.")

    def _search_queue_names(self):
        rds = self.redis_pool.get_connection()
        keys = rds.keys(TaskQueuePrefix + '*')
        return [k.decode('utf8') if isinstance(k, bytes) else k for k in keys]

    def _search_task(self):
        if self.is_task_group_queue_only:
            return self._search_group_task_queue()

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
        while not self.closed:
            self._check_cluster_status()
            st = time.time()
            if self.train_only:
                task = self._search_train_task()
            else:
                task = self._search_task()
            if task:
                if isinstance(task, Task):
                    task.setdefault("worker", self.worker_id)

                self.recorder.set_state(TaskReceiverState.Running)
                return task
            if not block:
                return None
            wait = sleep_time - time.time() + st
            if wait > 0:
                time.sleep(wait)
            self.register_worker()

    def task_iter(self, sleep_time: float = 2) -> typing.Iterable[Task]:
        while not self.closed:
            try:
                self._check_cluster_status()
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
                    if isinstance(task, Task):
                        task.setdefault("worker", self.worker_id)

                    self.recorder.set_state(TaskReceiverState.Running)
                    yield task
                else:
                    self.recorder.set_state(TaskReceiverState.Idle)

                wait = sleep_time - time.time() + st
                if wait > 0:
                    time.sleep(wait)
                self.register_worker()
                self.write_worker_state()
                self.exception_ts = -1
            except:
                time.sleep(1)
                logger.exception("get task err")
                self.exception_ts = time.time()
            finally:
                if self.exception_ts > 0 and time.time() - self.exception_ts > 900:
                    # 超过15分钟自动退出~
                    logger.warning(f'auto exit, catch error at:{self.exception_ts}')
                    self.close()
                    break

    def register_worker(self):
        if time.time() - self.register_time > 60:
            try:
                info = self._worker_info()
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] GPU free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
                conn = self.redis_pool.get_connection()
                conn.setex(self.worker_id, 300, json.dumps(info))

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

    def incr_train_concurrency(self, user_id: str):
        rds = self.redis_pool.get_connection()
        key = f'counter:train:{user_id}'
        v = rds.sadd(key, self.worker_id)
        logger.info(f'{key} + 1, current:{v}')
        rds.expire(key, 3 * 3600)

    def decr_train_concurrency(self, task: Task):
        if task.is_train:
            paralle_count = task.get('paralle_count', 0)
            if paralle_count > 0:
                user_id = task.user_id
                rds = self.redis_pool.get_connection()
                key = f'counter:train:{user_id}'
                v = rds.srem(key, self.worker_id)
                logger.info(f'{key} - 1, current:{v}')
                rds.expire(key, 3 * 3600)

    def can_exec_train_task(self, user_id: str, max_value: int):
        rds = self.redis_pool.get_connection()
        key = f'counter:train:{user_id}'
        members = rds.smembers(key)
        if len(members) < max_value:
            return True
        running_workers = self.get_all_workers()
        user_workers = set()

        logger.debug(f">>> {user_id} current workers:{'/n/t'.join(running_workers)}, /n/t max:{max_value}")
        flag = True
        members = [k.decode('utf8') if isinstance(k, bytes) else k for k in members]
        for worker_id in members:
            if worker_id in running_workers:
                user_workers.add(worker_id)
            if len(user_workers) >= max_value:
                flag = False
                break
        logger.info(f">>> {user_id} current trainning workers:{'/n/t'.join(user_workers)},/n/t max:{max_value}>{flag}")
        return flag

    def repush_task(self, task_id: str, queue_name: str, score: int):
        rds = self.redis_pool.get_connection()

        rds.zadd(queue_name, {
            task_id: score
        })

    def close(self):
        if self.closed:
            return

        self.closed = True
        self.timer.shutdown()
        self.redis_pool.close()
