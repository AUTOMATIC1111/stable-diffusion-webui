import torch


class DeviceProperties:
    type: str = "directml"
    name: str
    major: int = 0
    minor: int = 0
    total_memory: int
    multi_processor_count: int = 1

    def __init__(self, device: torch.device):
        self.name = torch.dml.get_device_name(device)
        self.total_memory = torch.dml.mem_get_info(device)[0]

    def __str__(self):
        return f"DeviceProperties(name='{self.name}', total_memory='{self.total_memory}')"

    def __repr__(self):
        return f"DeviceProperties(name='{self.name}', total_memory='{self.total_memory}')"
