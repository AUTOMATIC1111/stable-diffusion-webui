#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 5:50 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task.py
# @Software: Hifive
import abc
import json
import time
import typing
from enum import IntEnum
from datetime import datetime
from collections import UserDict, UserList

import math

from tools import try_deserialize_json


class SerializationObj:

    def to_dict(self):
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            try:
                if not name.startswith('_') and not callable(value):
                    if hasattr(value, 'to_dict'):
                        to_dict_func = getattr(value, 'to_dict')
                        if callable(to_dict_func):
                            value = value.to_dict()
                    pr[name] = value
            except:
                pass
        return pr


class Task(UserDict):

    def __init__(self, **kwargs):
        super(Task, self).__init__(None, **kwargs)
        if 'create_at' not in self or self['create_at'] < 1:
            self['create_at'] = int(time.time())

    @property
    def id(self):
        task_id = self.get("task_id")
        return task_id

    @property
    def user_id(self):
        return self.get("user_id")

    @property
    def task_type(self):
        return TaskType(self['task_type'])

    @property
    def minor_type(self):
        return self.get('minor_type', 0)

    @property
    def create_at(self):
        return self.get('create_at')

    @property
    def is_train(self):
        return TaskType.Train == self.task_type

    def stop_receiver(self):
        return (self.is_train and self.minor_type > 1)

    @property
    def sd_model_path(self):
        return self.get('base_model_path')

    @property
    def model_hash(self):
        return self.get('model_hash')

    @property
    def lora_models(self):
        return self.get('lora_models')

    @classmethod
    def from_json_str(cls, json_str: str):
        meta = try_deserialize_json(json_str)
        if meta:
            return cls(**meta)

    def valid(self) -> typing.Tuple[bool, str]:
        if not self.id:
            return False, "cannot found task id"
        if not self.get('user_id'):
            return False, f"cannot found user id, task id: {self.id}"
        if not self.get('task_type'):
            return False, f"cannot found task type, task id: {self.id}"
        try:
            TaskType(self['task_type'])
        except:
            return False, f"valid task type:{self['task_type']}, task id: {self.id}"

        return True, ""

    def desc(self) -> str:
        task_id = self.id or 'unknown task'
        return f'taskId:{task_id}, type:{self.task_type.name}'

    def json(self) -> str:
        return json.dumps(dict(self.items()))

    def value(self, key, default=None, requires=False):
        if requires:
            return self[key]
        else:
            return self.get(key, default=default)

    def to_dict(self):
        return dict(self)


class TaskType(IntEnum):
    Txt2Image = 1
    Image2Image = 2
    Extra = 3
    Train = 4
    Tagger=5

class TaskStatus(IntEnum):
    Waiting = 0
    Prepare = 1
    Ready = 2
    Running = 3
    Uploading = 4
    TrainCompleted = 9
    Finish = 10
    Failed = -1


class TrainEpoch(SerializationObj):

    def __init__(self, epoch, loss):
        self.epoch = epoch
        self.loss = loss
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TrainEpochLog(UserList, SerializationObj):

    def append(self, item):
        if not isinstance(item, TrainEpoch):
            raise TypeError
        self.data.append(item.to_dict())

    def insert(self, i, item):
        if not isinstance(item, TrainEpoch):
            raise TypeError
        self.data.insert(i, item.to_dict())

    def extend(self, other):
        if isinstance(other, TrainEpochLog):
            self.data.extend(other.data)
        else:
            self.data.extend(other)

    def to_dict(self):
        return self.data


class TrainTaskInfo(SerializationObj):

    def __init__(self):
        self.epoch = TrainEpochLog()

    def add_epoch_log(self, epoch: TrainEpoch):
        self.epoch.append(epoch)

    def format_epoch_log(self):
        lines = []
        for item in self.epoch:
            loss = item['loss']
            epoch = item['epoch']
            time = item['time']
            lines.append(f'[{time}] > epoch:{epoch}, loss:{loss}')
        return '\n'.join(lines)


class TaskProgress(SerializationObj):

    def __init__(self, task: Task):
        self.status = TaskStatus.Waiting
        self.task_desc = 'waiting'
        self.task = task
        self._result = None
        self.task_progress = 0
        self.eta_relative = None
        self.train = TrainTaskInfo()
        self.preview = None

    @property
    def completed(self):
        return self.status == TaskStatus.Finish or self.status == TaskStatus.Failed

    @property
    def result(self):
        return self._result

    def pre_task_completed(self):
        return self.completed or self.status >= TaskStatus.Uploading

    def set_status(self, status: TaskStatus, desc: str):
        self.task_desc = desc
        self.status = status

    def set_finish_result(self, r: typing.Any, is_train_task=False):
        self._result = r
        self.status = TaskStatus.Finish if not is_train_task else TaskStatus.TrainCompleted
        self.task_desc = 'ok'
        self.task_progress = 100 if not is_train_task else 99

    def update_seed(self, seed, sub_seed):
        if isinstance(self.task, Task):
            self.task['all_seed'] = seed
            self.task['all_sub_seed'] = sub_seed

    @classmethod
    def new_failed(cls, task: Task, desc: str, trace: str = None):
        task.update(
            {
                "end_at": int(time.time()) - task.create_at,
            }
        )
        p = cls(task)
        p.status = TaskStatus.Failed
        p.task_desc = desc
        p.task_progress = 0
        p.trace = trace
        return p

    @classmethod
    def new_running(cls, task: Task, desc: str, progress=0):
        p = cls(task)
        p.status = TaskStatus.Running
        p.task_desc = desc
        p.task_progress = progress
        return p

    @classmethod
    def new_ready(cls, task: Task, desc: str):
        p = cls(task)
        p.status = TaskStatus.Ready
        p.task_desc = desc
        return p

    @classmethod
    def new_prepare(cls, task: Task, desc: str):
        task.update(
            {
                "latency": int(time.time()) - task.create_at,
            }
        )
        p = cls(task)
        p.status = TaskStatus.Prepare
        p.task_desc = desc
        return p

    @classmethod
    def new_finish(cls, task: Task, result: typing.Any, is_train_task=False):
        task.update(
            {
                "end_at": int(time.time())
            }
        )
        p = cls(task)
        p.set_finish_result(result, is_train_task)
        return p
