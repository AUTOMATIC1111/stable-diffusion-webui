from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict

import torch

from modules import mem_usage

log = logging.getLogger(__name__)


class MemUsageMonitor(threading.Thread):
    run_flag: threading.Event
    device: torch.device
    api: mem_usage.MemUsageAPI | None

    def __init__(self, *, name="MemMon", device: torch.device, poll_rate: float = 8):
        super().__init__(name=name, daemon=True)
        self.poll_rate = poll_rate
        self.device = device
        self.run_flag = threading.Event()

        try:
            if device.type == "cuda":
                self.api = mem_usage.CudaMemUsageAPI(device)
            elif device.type == "mps":
                self.api = mem_usage.MPSMemUsageAPI(device)
            else:
                raise ValueError(f"Unsupported device type: {device.type}")
        except Exception as e:
            log.warning(f"Memory monitor disabled: {e}", exc_info=True)
            self.api = None

    @property
    def disabled(self) -> bool:
        return self.api is None

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()
            self.api.reset_peak()

            if self.poll_rate <= 0:
                self.run_flag.clear()
                continue

            while self.run_flag.is_set():
                self.api.read()  # This updates peak values, even if we don't use the retval here
                time.sleep(1 / self.poll_rate)

    def monitor(self) -> None:
        self.run_flag.set()

    def read(self) -> dict[str, int | None]:
        if not self.api:
            return {}
        stats = self.api.get_stats()
        stats.update({f"min_{k}": v for k, v in self.api.min.items()})
        stats.update({f"max_{k}": v for k, v in self.api.max.items()})
        return stats

    def stop(self) -> None:
        self.run_flag.clear()
