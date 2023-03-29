#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/29 4:25 PM
# @Author  : wangdongming
# @Site    : 
# @File    : model_hist.py
# @Software: Hifive
import threading
from collections import OrderedDict


class ModelHistory:

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._history = OrderedDict()
        self._locker = threading.RLock()

    @property
    def length(self):
        return len(self._history)

    def history(self, reverse=False):
        if not self._history:
            return []
        keys = list(self._history.keys())
        if reverse:
            keys = keys[::-1]
        return keys

    def switch_model(self, model_hash: str):
        if model_hash in self._history or not model_hash:
            return
        self._push(model_hash)

    def _pop(self):
        if self.length >= self.capacity:
            with self._locker:
                while self.length >= self.capacity:
                    self._history.popitem(last=False)

    def _push(self, model_hash: str):
        self._pop()
        if self.length < self.capacity and model_hash not in self._history:
            with self._locker:
                if self.length < self.capacity and model_hash not in self._history:
                    self._history[model_hash] = True
