import threading
import time
from collections import defaultdict

import torch


class MemUsageMonitor(threading.Thread):
    run_flag: threading.Event
    device: torch.device
    disabled: bool
    data: defaultdict

    def __init__(self, *, name="MemMon", device: torch.device, poll_rate: float = 8):
        super().__init__(name=name, daemon=True)
        self.poll_rate = poll_rate
        self.device = device
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            self.cuda_mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.poll_rate)

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

    def monitor(self) -> None:
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self) -> None:
        self.run_flag.clear()
