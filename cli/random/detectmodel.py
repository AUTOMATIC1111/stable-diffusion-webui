#!/bin/env python
"""
Detect model type

Works for v1 and v2-base (EPS models), both standard inference and inpainting

But looking at model dumps between EPS and V type models, its only about parametrization, there are no differences in actual model (its just weighted differently without any structural difference)
So i don't see easy way to auto-detect if model should be run in `EPS` or `V` mode
Only difference are some calculations in `ldm/models/diffusion/ddpm.py` and by then we already need to know which code-path to trigger
(maaaybe there could be a way by looking at some cherry-picked base tensors min/max range, but I dont see that as reliable)
"""

import os
import sys

import torch

def signature(model):
    if model is None:
        return None
    try:
        size = model['state_dict']['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'].shape[1]
        unet = model['state_dict']['model.diffusion_model.input_blocks.0.0.weight'].shape[1]
    except:
        return 'unknown'
    guess = 'v1' if size == 768 else 'v2' # 768 for v1 and 1024 for v2
    guess += '-inference' if unet == 4 else '-inpainting' # inference models have shorter inputs, 4 for inference 9 for inpainting
    return guess

def load(file: str):
    try:
        model = torch.load(file, map_location='cpu')
        return model
    except Exception as err:
        print(f"Error loading {f}: {err}")

if __name__ == "__main__":
    sys.argv.pop(0)
    for f in sys.argv:
        if os.path.isfile(f):
            print(f"Model {f} is of type {signature(load(f))}")
        else:
            print(f"{f} is not a file")
