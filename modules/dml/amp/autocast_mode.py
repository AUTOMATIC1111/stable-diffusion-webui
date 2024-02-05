import importlib
from typing import Any, Optional
import torch


ops = ["torch.Tensor.__matmul__", "torch.addbmm", "torch.addmm", "torch.addmv", "torch.addr", "torch.baddbmm", "torch.bmm", "torch.chain_matmul", "torch.linalg.multi_dot", "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d", "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d", "torch.nn.GRUCell", "torch.nn.functional.linear", "torch.nn.LSTMCell", "torch.matmul", "torch.mm", "torch.mv", "torch.prelu", "torch.nn.RNNCell", "torch.embedding"]
supported_cast_pairs = {
    torch.float16: (torch.float32,),
    torch.float32: (torch.float16,),
}


def forward(op, args: tuple, kwargs: dict):
    if not torch.dml.is_autocast_enabled:
        return op(*args, **kwargs)
    args = list(map(cast, args))
    for kwarg in kwargs:
        kwargs[kwarg] = cast(kwargs[kwarg])
    return op(*args, **kwargs)


def cast(tensor: torch.Tensor):
    if not torch.is_tensor(tensor):
        return tensor
    dtype: torch.dtype = tensor.dtype
    if dtype not in supported_cast_pairs or (torch.dml.autocast_gpu_dtype != dtype and torch.dml.autocast_gpu_dtype not in supported_cast_pairs[dtype]):
        return tensor
    return tensor.type(torch.dml.autocast_gpu_dtype)


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
        setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: forward(op, args, kwargs))


for o in ops:
    cond(o)


class autocast:
    prev: bool

    fast_dtype: torch.dtype = torch.float16
    prev_fast_dtype: torch.dtype
    def __init__(self, dtype: Optional[torch.dtype] = torch.float16):
        self.fast_dtype = dtype

    def __enter__(self):
        self.prev = torch.dml.is_autocast_enabled
        self.prev_fast_dtype = torch.dml.autocast_gpu_dtype
        torch.dml.is_autocast_enabled = True
        torch.dml.autocast_gpu_dtype = self.fast_dtype

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        torch.dml.is_autocast_enabled = self.prev
        torch.dml.autocast_gpu_dtype = self.prev_fast_dtype
