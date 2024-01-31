import os
from typing import List

import numpy as np
import torch
from safetensors.torch import load_file
import onnx
from onnx import numpy_helper


def merge_loras(loras: List[str], scales: List[str]) -> dict:
    refit_dict = {}
    for lora, scale in zip(loras, scales):
        lora_dict = load_file(lora)
        for k, v in lora_dict.items():
            if k in refit_dict:
                refit_dict[k] += scale * v
            else:
                refit_dict[k] = scale * v
    return refit_dict


def apply_loras(base_path: str, loras: List[str], scales: List[str]) -> dict:
    refit_dict = merge_loras(loras, scales)
    base = onnx.load(base_path)
    onnx_opt_dir = os.path.dirname(base_path)

    def convert_int64(arr):
        if len(arr.shape) == 0:
            return np.array([np.int32(arr)])
        return arr

    for initializer in base.graph.initializer:
        if initializer.name not in refit_dict:
            continue

        wt = refit_dict[initializer.name]
        initializer_data = numpy_helper.to_array(
            initializer, base_dir=onnx_opt_dir
        ).astype(np.float16)
        delta = torch.tensor(initializer_data).to(wt.device) + wt

        refit_dict[initializer.name] = delta.contiguous()

    return refit_dict
