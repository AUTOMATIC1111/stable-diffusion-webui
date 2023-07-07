#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 2:31 PM
# @Author  : wangdongming
# @Site    :
# @File    : dumper_test.py
# @Software: Hifive
import os
import time
import unittest
from worker.task import TaskProgress, Task
from worker.dumper import dumper
from tools.environment import Env_MgoHost, Env_MgoPass, Env_MgoUser


class TestMgoDumper(unittest.TestCase):

    def test_dumper(self):
        task_ids = ["test-1", "test-2"]
        for p in range(0, 110, 10):
            for task_id in task_ids:
                t = Task(user_id='unittest', task_type=1, task_id=task_id)
                progress = TaskProgress(t)
                progress.task_progress = p
                dumper.dump_task_progress(progress)
            print(f"current progress:{p}")
            time.sleep(2)
        time.sleep(11)
        dumper.stop()


if __name__ == '__main__':
    unittest.main()