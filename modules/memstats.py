import os
import psutil
import torch
from modules import shared, errors

fail_once = False

def memory_stats():
    global fail_once # pylint: disable=global-statement
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
        if not fail_once:
            shared.log.error('Memory stats: {e}')
            errors.display(e, 'Memory stats')
            fail_once = True
        mem.update({ 'ram': str(e) })
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
