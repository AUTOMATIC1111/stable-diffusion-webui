import importlib
from typing import Any, Optional
import torch

ops = ["torch.Tensor.__matmul__", "torch.addbmm", "torch.addmm", "torch.addmv", "torch.addr", "torch.baddbmm", "torch.bmm", "torch.chain_matmul", "torch.linalg.multi_dot", "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d", "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d", "torch.nn.GRUCell", "torch.nn.functional.linear", "torch.nn.LSTMCell", "torch.matmul", "torch.mm", "torch.mv", "torch.prelu", "torch.nn.RNNCell"]

def pre_forward(forward, args, kwargs):
    if not torch.dml.is_autocast_enabled():
        return forward(*args, **kwargs)
    args = list(map(cast, args))
    for keyword in kwargs:
        kwargs[keyword] = cast(kwargs[keyword])
    return forward(*args, **kwargs)

def cast(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.type(torch.dml.get_autocast_gpu_dtype())

def cond(op: str):
    if isinstance(op, str):
        func_path = op.split('.')
        for i in range(len(func_path)-1, -1, -1):
            try:
                resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                break
            except ImportError:
                pass
        for attr_name in func_path[i:-1]:
            resolved_obj = getattr(resolved_obj, attr_name)
        op = getattr(resolved_obj, func_path[-1])
        setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: pre_forward(op, args, kwargs))

for op in ops:
    cond(op)

class autocast:
    def __init__(self, dtype: Optional[torch.dtype] = None):
        self.fast_dtype = dtype or torch.dml.get_autocast_gpu_dtype()

    def __enter__(self):
        self.prev = torch.dml.is_autocast_enabled()
        self.prev_fastdtype = torch.dml.get_autocast_gpu_dtype()
        torch.dml.set_autocast_enabled(True)
        torch.dml.set_autocast_gpu_dtype(self.fast_dtype)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        torch.dml.set_autocast_enabled(self.prev)
        torch.dml.set_autocast_gpu_dtype(self.prev_fastdtype)
