#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 5:17 PM
# @Author  : wangdongming
# @Site    : 
# @File    : processor.py
# @Software: Hifive

import time
import math
from threading import Thread
from queue import Queue as ThreadQueue, Empty
from multiprocessing import Process, Queue as ProcessQueue, cpu_count


class MultiWorker(object):
    """
    多任务处理器
    """
    queue_type = ThreadQueue
    worker_type = Thread
    max_worker_count = 10
    max_queue_size = 10

    @property
    def queue(self):
        if self._queue is None:
            self._queue = self.queue_type(maxsize=self.max_queue_size)
        return self._queue

    def __init__(self, generator, target, worker_count):
        """
        提供一个任务输入生成器及任务函数，生成指定数量的worker，来并行处理任务
        @param generator: 任务生成器，每次迭代返回任务输入所需参数
        @param target: 任务处理函数，接收任务生成器传入的参数执行处理任务。
        @param worker_count: 希望的worker数量，最多不超过max_worker_count
        """
        if worker_count is not None:
            self.max_worker_count = min(worker_count, self.max_worker_count)
        self.generator = generator
        self.target = target
        self.start = False
        self._queue = None

    def run(self):
        """
        任务调度开始
        @return:
        """
        self.start = True
        workers = list()

        # 创建指定数量的worker， 设置为守护模式
        for i in range(self.max_worker_count):
            worker = self.worker_type(target=self.process)
            worker.daemon = True
            workers.append(worker)
            worker.start()

        current_progress = 0
        counter = 0

        # 投放任务
        for task in self.generator:
            if isinstance(task, dict):
                if "count" in task:
                    count = task['count']
                    if 'counter' in task:
                        c = task['counter']
                    else:
                        c = counter
                    progress = int(math.ceil(c * 100 / count))
                    if progress > current_progress:
                        current_progress = progress
                        self.progress(current_progress)
                if "task" in task:
                    task = task['task']
            counter += 1
            self.queue.put(task)

        # 等待任务完成
        while not self.queue.empty() and any(w.is_alive() for w in workers):
            time.sleep(0.3)

        # 等待最后一波任务完成。
        self.start = False
        for worker in workers:
            if worker.is_alive():
                worker.join()
        self.completed()

    def process(self):
        while self.start:
            try:
                self.target(*self.queue.get_nowait())
            except Empty:
                time.sleep(0.1)
                continue
            except Exception as ex:
                if self.error(ex):
                    continue

    def progress(self, p: int):
        print(f"progress:{p}%")

    def completed(self):
        print("all task finished.")

    def error(self, exception):
        print("unhandle error:" + str(exception))
        raise exception


class MultiThreadWorker(MultiWorker):
    pass


class MultiProcessWorker(MultiWorker):
    max_worker_count = cpu_count()
    queue_type = ProcessQueue
    worker_type = Process

