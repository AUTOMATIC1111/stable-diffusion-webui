import threading
import time
from collections import defaultdict
import torch
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
        if not torch.cuda.is_available():
            #torch.cuda.is_available() reports False when using IPEX.
            if shared.cmd_opts.use_ipex:
                self.cuda_mem_get_info()
                torch.xpu.memory_stats("xpu")
            else:
                self.disabled = True
        else:
            try:
                self.cuda_mem_get_info()
                torch.cuda.memory_stats(self.device)
            except Exception:
                self.disabled = True

    def cuda_mem_get_info(self):
        if shared.cmd_opts.use_ipex:
            #-128MB for the OS and other.
            return [(torch.xpu.get_device_properties("xpu").total_memory - torch.xpu.memory_allocated() - (128*1024*1024)), torch.xpu.get_device_properties("xpu").total_memory]
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

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total
            try:
                if shared.cmd_opts.use_ipex:
                    torch_stats = torch.xpu.memory_stats("xpu")
                else:
                    torch_stats = torch.cuda.memory_stats(self.device)
                self.data["active"] = torch_stats["active.all.current"]
                self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
                self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
                self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
                self.data["system_peak"] = total - self.data["min_free"]
            except:
                self.disabled = True
        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
