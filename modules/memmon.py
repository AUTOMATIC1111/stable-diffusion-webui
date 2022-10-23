import threading
import time
from collections import defaultdict

import torch


class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        print("Init memmmon")
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            torch.cuda.mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def run(self):
        if self.disabled:
            print("Memmmon disabled")
            return
        print("Running memmon")
        logged = False
        while True:
            #self.run_flag.wait()
            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            self.data["min_free"] = torch.cuda.mem_get_info()[0]
            allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
            cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
            print('Monitoring memory:')
            print('Allocated:', allocated, 'GB')
            print('Reserved:   ', cached, 'GB')
            logged = True
            while True:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Additional Info when using cuda
                if device.type == 'cuda':
                    last_alloc = allocated
                    last_cached = cached
                    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
                    cached = round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)
                    if last_alloc != allocated or last_cached != cached:
                        logged = False
                    else:
                        if not logged:
                            print('Memory changed:')
                            print('Allocated:', allocated, 'GB')
                            print('Reserved:   ', cached, 'GB')
                            logged = True

                free, total = torch.cuda.mem_get_info()  # calling with self.device errors, torch bug?
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = torch.cuda.mem_get_info()
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
