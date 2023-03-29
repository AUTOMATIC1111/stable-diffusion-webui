#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 5:50 PM
# @Author  : wangdongming
# @Site    : 
# @File    : task.py
# @Software: Hifive
import abc
import typing
from enum import IntEnum

from tools import try_deserialize_json


class Task:

    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.task_type = kwargs.get('task_type')
        self.user_id = kwargs.get('user_id')
        self.sd_model = kwargs.get('sd_model')


    @classmethod
    def from_json_str(cls, json_str: str):
        meta = try_deserialize_json(json_str)
        if meta:
            return cls(**meta)

    def valid(self) -> typing.Tuple[bool, str]:
        if not self.id:
            return False, "cannot found task id"
        if not self.user_id:
            return False, f"cannot found user id, task id: {self.id}"
        if not self.task_type:
            return False, f"cannot found task type, task id: {self.id}"
        return True, ""


class TaskStatus(IntEnum):
    Waiting = 0
    Ready = 1
    Running = 2
    Finsh = 10
    Failed = -1


class TaskProgress(Task):

    def __init__(self, **kwargs):
        super(TaskProgress, self).__init__(**kwargs)
        self.status = TaskStatus.Waiting
        self.task_desc = 'waiting'

    @property
    def task_finished(self):
        return self.status == TaskStatus.Waiting or self.status == TaskStatus.Failed 


class TaskHandler:

    def __init__(self):
        self.task_type = 0

    @abc.abstractmethod
    def _exec(self, task: Task):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_task_status(self, task: Task):
        raise NotImplementedError

    def do(self, task: Task):
        ok, msg = task.valid()
        if not ok:
            task.task_desc = msg
            task.status = TaskStatus.Failed
            self._set_task_status(task)
        else:
            pass
        # TODO:
                
