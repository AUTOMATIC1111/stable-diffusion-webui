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
import traceback
import typing

from loguru import logger
from enum import IntEnum
from collections import UserDict
from tools import try_deserialize_json
from tools.encryptor import string_to_hex


class Task(UserDict):

    def __init__(self, **kwargs):
        super(Task, self).__init__(None, **kwargs)
        if 'create_at' not in self:
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


class TaskType(IntEnum):
    Txt2Image = 1
    Image2Image = 2
    Extra = 3


class TaskStatus(IntEnum):
    Waiting = 0
    Prepare = 1
    Ready = 2
    Running = 3
    Finish = 10
    Failed = -1


class TaskProgress:

    def __init__(self, task: Task):
        self.status = TaskStatus.Waiting
        self.task_desc = 'waiting'
        self.task = task
        self._result = None
        self.task_progress = 0

    @property
    def completed(self):
        return self.status == TaskStatus.Finish or self.status == TaskStatus.Failed

    @property
    def result(self):
        return self._result

    def set_status(self, status: TaskStatus, desc: str):
        self.task_desc = desc
        self.status = status

    def set_finish_result(self, r: typing.Any):
        self._result = r
        self.status = TaskStatus.Finish
        self.task_desc = 'ok'

    def to_dict(self):
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            try:
                if not name.startswith('_') and not callable(value):
                    pr[name] = value
            except:
                pass
        return pr

    @classmethod
    def new_failed(cls, task: Task, desc: str):
        p = cls(task)
        p.status = TaskStatus.Failed
        p.task_desc = desc
        p.task_progress = 0
        return p

    @classmethod
    def new_running(cls, task: Task, desc: str):
        p = cls(task)
        p.status = TaskStatus.Running
        p.task_desc = desc
        return p

    @classmethod
    def new_ready(cls, task: Task, desc: str):
        p = cls(task)
        p.status = TaskStatus.Ready
        p.task_desc = desc
        return p

    @classmethod
    def new_prepare(cls, task: Task, desc: str):
        p = cls(task)
        p.status = TaskStatus.Prepare
        p.task_desc = desc
        return p

    @classmethod
    def new_finish(cls, task: Task, result: typing.Any):
        p = cls(task)
        p.status = TaskStatus.Finish
        p.task_desc = 'ok'
        p._result = result
        return p


class TaskHandler:

    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def handle_task_type(self):
        return self.task_type

    @abc.abstractmethod
    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        raise NotImplementedError

    def _set_task_status(self, p: TaskProgress):
        logger.info(f">>> task:{p.task.desc()}, status:{p.status.name}, desc:{p.task_desc}")

    def do(self, task: Task):
        ok, msg = task.valid()
        if not ok:
            p = TaskProgress.new_failed(task, msg)
            self._set_task_status(p)
        else:
            try:
                p = TaskProgress.new_ready(task, msg)
                self._set_task_status(p)
                for progress in self._exec(task):
                    self._set_task_status(progress)
            except Exception:
                msg = traceback.format_exc()
                p = TaskProgress.new_failed(task, msg)
                self._set_task_status(p)

    def close(self):
        pass

    def set_failed(self, task: Task, desc: str):
        p = TaskProgress.new_failed(task, desc)
        self._set_task_status(p)

    def __call__(self, task: Task):
        return self.do(task)
