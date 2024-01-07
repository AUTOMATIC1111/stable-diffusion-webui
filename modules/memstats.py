import os
import psutil
import torch
from modules import shared

def memory_stats():
    def gb(val: float):
        return round(val / 1024 / 1024 / 1024, 2)

    mem = {}
    try:
        process = psutil.Process(os.getpid())
        res = process.memory_info()
        ram_total = 100 * res.rss / process.memory_percent()
        ram = { 'used': gb(res.rss), 'total': gb(ram_total) }
        mem.update({ 'ram': ram })
    except Exception as e:
        mem.update({ 'ram': e })
    try:
        s = torch.cuda.mem_get_info()
        gpu = { 'used': gb(s[1] - s[0]), 'total': gb(s[1]) }
        s = dict(torch.cuda.memory_stats())
        if s['num_ooms'] > 0:
            shared.state.oom = True
        mem.update({
            'gpu': gpu,
            'retries': s['num_alloc_retries'],
            'oom': s['num_ooms']
        })
        return mem
    except Exception:
        pass
    return mem
