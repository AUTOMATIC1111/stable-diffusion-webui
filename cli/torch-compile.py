#!/usr/bin/env python
# pylint: disable=cell-var-from-loop
"""
Test Torch Dynamo functionality and backends
"""
import json
import warnings

import numpy as np
import torch
from torchvision.models import resnet18


print('torch:', torch.__version__)
try:
    # must be imported explicitly or namespace is not found
    import torch._dynamo as dynamo # pylint: disable=ungrouped-imports
except Exception as err:
    print('torch without dynamo support', err)


N_ITERS = 20
torch._dynamo.config.verbose=True # pylint: disable=protected-access
warnings.filterwarnings('ignore', category=UserWarning) # disable those for now as many backends reports tons
# torch.set_float32_matmul_precision('high') # enable to test in fp32


def timed(fn): # returns the result of running `fn()` and the time it took for `fn()` to run in ms using CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )


def init_model():
    return resnet18().to(torch.float32).cuda()


def evaluate(mod, val):
    return mod(val)


if __name__ == '__main__':
    # first pass, dynamo is going to be slower as it compiles
    model = init_model()
    inp = generate_data(16)[0]

    # repeat test
    results = {}
    times = []
    print('eager initial eval:', timed(lambda: evaluate(model, inp))[1])
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        _res, time = timed(lambda: evaluate(model, inp))
        times.append(time)
    results['default'] = np.median(times)

    print('dynamo available backends:', dynamo.list_backends())
    for backend in dynamo.list_backends():
        try:
            # required before changing backends
            torch._dynamo.reset() # pylint: disable=protected-access
            eval_dyn = dynamo.optimize(backend)(evaluate)
            print('dynamo initial eval:', backend, timed(lambda: eval_dyn(model, inp))[1])
            times = []
            for i in range(N_ITERS):
                inp = generate_data(16)[0]
                _res, time = timed(lambda: eval_dyn(model, inp))
                times.append(time)
            results[backend] = np.median(times)
        except Exception as err:
            lines = str(err).split('\n')
            print('dyanmo backend failed:', backend, lines[0]) # print just first error line as backtraces can be quite long
            results[backend] = 'error'

    # print stats
    print(json.dumps(results, indent = 4))


"""
Reference: <https://github.com/pytorch/pytorch/blob/4f4b62e4a255708e928445b6502139d5962974fa/docs/source/dynamo/get-started.rst>
Training & Inference backends:
    dynamo.optimize("inductor") - Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton kernels
    dynamo.optimize("aot_nvfuser") - nvFuser with AotAutograd
    dynamo.optimize("aot_cudagraphs") - cudagraphs with AotAutograd
Inference-only backends:
    dynamo.optimize("ofi") - Uses Torchscript optimize_for_inference
    dynamo.optimize("fx2trt") - Uses Nvidia TensorRT for inference optimizations
    dynamo.optimize("onnxrt") - Uses ONNXRT for inference on CPU/GPU
"""
