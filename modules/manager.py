#
# based on forge's work from https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/modules_forge/main_thread.py
#
# Original author comment:
#  This file is the main thread that handles all gradio calls for major t2i or i2i processing.
#  Other gradio calls (like those from extensions) are not influenced.
#  By using one single thread to process all major calls, model moving is significantly faster.
#
# 2024/09/28 classified,

import random
import string
import threading
import time

from collections import OrderedDict


class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TaskManager:
    last_exception = None
    pending_tasks = []
    finished_tasks = OrderedDict()
    lock = None
    running = False

    def __init__(self):
        self.lock = threading.Lock()

    def work(self, task):
        try:
            task.result = task.func(*task.args, **task.kwargs)
        except Exception as e:
            task.exception = e
            self.last_exception = e


    def stop(self):
        self.running = False


    def main_loop(self):
        self.running = True
        while self.running:
            time.sleep(0.01)
            if len(self.pending_tasks) > 0:
                with self.lock:
                    task = self.pending_tasks.pop(0)

                self.work(task)

                self.finished_tasks[task.task_id] = task


    def push_task(self, func, *args, **kwargs):
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            task_id = args[0]
        else:
            task_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        task = Task(task_id=task_id, func=func, args=args, kwargs=kwargs, result=None, exception=None)
        self.pending_tasks.append(task)

        return task.task_id


    def run_and_wait_result(self, func, *args, **kwargs):
        current_id = self.push_task(func, *args, **kwargs)

        while True:
            time.sleep(0.01)
            if current_id in self.finished_tasks:
                finished = self.finished_tasks.pop(current_id)
                if finished.exception is not None:
                    raise finished.exception

                return finished.result


task = TaskManager()
