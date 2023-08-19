import os
import torch
from openvino.frontend.pytorch.torchdynamo.execute import execute
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is None:
        os.environ.setdefault("OPENVINO_TORCH_BACKEND_DEVICE", "GPU")

    model = make_fx(subgraph)(*example_inputs)
    with torch.no_grad():
        model.eval()
    partitioner = Partitioner()
    compiled_model = partitioner.make_partitions(model)

    def _call(*args):
        res = execute(compiled_model, *args, executor="openvino")
        return res
    return _call
