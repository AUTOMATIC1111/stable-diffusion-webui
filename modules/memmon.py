import threading
import time
from collections import defaultdict
import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from modules import shared


class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts
        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)
        if not torch.cuda.is_available() or not shared.cmd_opts.use_ipex:
            self.disabled = True
        else:
            if shared.cmd_opts.use_ipex:
                try:
                    self.cuda_mem_get_info()
                    torch.cuda.memory_stats("xpu")
                except Exception as e:  # AMD or whatever
                    print(f"Torch exception: {e}")
                    self.disabled = True

            else:
                try:
                    self.cuda_mem_get_info()
                    torch.cuda.memory_stats(self.device)
                except Exception as e:  # AMD or whatever
                    print(f"Torch exception: {e}")
                    self.disabled = True

    def cuda_mem_get_info(self):
        if shared.cmd_opts.use_ipex:
            return [(torch.xpu.get_device_properties("xpu").total_memory - torch.xpu.memory_allocated()), torch.xpu.get_device_properties("xpu").total_memory]
        else:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return
        while True:
            self.run_flag.wait()
            if shared.cmd_opts.use_ipex:
                torch.xpu.reset_peak_memory_stats()
            else:
                torch.cuda.reset_peak_memory_stats()
            self.data.clear()
            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue
            self.data["min_free"] = self.cuda_mem_get_info()[0]
            while self.run_flag.is_set():
                free, _total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)
                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))
        print(self, 'raw torch memory stats:')
        if shared.cmd_opts.use_ipex:
            tm = torch.xpu.memory_stats("xpu")
        else:
            tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))
        if shared.cmd_opts.use_ipex:
            print(torch.xpu.memory_summary())
        else:
            print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            if shared.cmd_opts.use_ipex:
                torch_stats = torch.xpu.memory_stats("xpu")
            else:
                torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
