from __future__ import annotations

import torch


class MemUsageAPI:
    def __init__(self, device: torch.device):
        self.device = device
        self.min = {}
        self.max = {}

    def reset_peak(self) -> None:
        self.min.clear()
        self.max.clear()

    def update_peak(self, stats: dict[str, int | None]) -> None:
        for k, v in stats.items():
            if v is None:
                continue
            if k not in self.min or v < self.min[k]:
                self.min[k] = v
            if k not in self.max or v > self.max[k]:
                self.max[k] = v

    def get_stats(self) -> dict[str, int | None]:
        """
        Get raw stats from the underlying device.
        """
        raise NotImplementedError("get_stats() not implemented")

    def read(self) -> dict[str, int | None]:
        """
        Get stats from the underlying device, update peak values, and return data with min_ and max_ merged in.
        """
        stats = self.get_stats()
        self.update_peak(stats)
        stats.update({f"min_{k}": v for k, v in self.min.items()})
        stats.update({f"max_{k}": v for k, v in self.max.items()})
        return stats


class CudaMemUsageAPI(MemUsageAPI):
    def reset_peak(self):
        super().reset_peak()
        torch.cuda.reset_peak_memory_stats(self.device)

    def get_stats(self) -> dict[str, int | None]:
        free, total = torch.cuda.mem_get_info(self.device)
        torch_stats = torch.cuda.memory_stats(self.device)
        return {
            "free": free,
            "total": total,
            "active": torch_stats["active.all.current"],
            "active_peak": torch_stats["active_bytes.all.peak"],
            "reserved": torch_stats["reserved_bytes.all.current"],
            "reserved_peak": torch_stats["reserved_bytes.all.peak"],
            "system_peak": total - self.min.get("free", total),
        }
