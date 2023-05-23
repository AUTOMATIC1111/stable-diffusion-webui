#!/usr/bin/env python
"""
generic helper methods
"""

import os
import string
import logging
import warnings

log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level = logging.INFO, format = log_format)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
log = logging.getLogger("sd")


def set_logfile(logfile):
    fh = logging.FileHandler(logfile)
    formatter = logging.Formatter(log_format)
    fh.setLevel(log.getEffectiveLevel())
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info({ 'log file': logfile })


def safestring(text: str):
    lines = []
    for line in text.splitlines():
        lines.append(line.translate(str.maketrans('', '', string.punctuation)).strip())
    res = ', '.join(lines)
    return res[:1000]


def get_memory():
    def gb(val: float):
        return round(val / 1024 / 1024 / 1024, 2)
    mem = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram = { 'free': gb(ram_total - res.rss), 'used': gb(res.rss), 'total': gb(ram_total) }
        mem.update({ 'ram': ram })
    except Exception as e:
        mem.update({ 'ram': e })
    try:
        import torch
        if torch.cuda.is_available():
            s = torch.cuda.mem_get_info()
            gpu = { 'free': gb(s[0]), 'used': gb(s[1] - s[0]), 'total': gb(s[1]) }
            s = dict(torch.cuda.memory_stats('cuda'))
            allocated = { 'current': gb(s['allocated_bytes.all.current']), 'peak': gb(s['allocated_bytes.all.peak']) }
            reserved = { 'current': gb(s['reserved_bytes.all.current']), 'peak': gb(s['reserved_bytes.all.peak']) }
            active = { 'current': gb(s['active_bytes.all.current']), 'peak': gb(s['active_bytes.all.peak']) }
            inactive = { 'current': gb(s['inactive_split_bytes.all.current']), 'peak': gb(s['inactive_split_bytes.all.peak']) }
            events = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
            mem.update({
                'gpu': gpu,
                'gpu-active': active,
                'gpu-allocated': allocated,
                'gpu-reserved': reserved,
                'gpu-inactive': inactive,
                'events': events,
            })
    except:
        pass
    return Map(mem)


class Map(dict): # pylint: disable=C0205
    __slots__ = ('__dict__') # pylint: disable=C0325
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = Map(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = Map(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v
    def __convert(self, v):
        for elem in range(0, len(v)): # pylint: disable=consider-using-enumerate
            if isinstance(v[elem], dict):
                v[elem] = Map(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])
    def __getattr__(self, attr):
        return self.get(attr)
    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})
    def __delattr__(self, item):
        self.__delitem__(item)
    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


if __name__ == "__main__":
    pass
