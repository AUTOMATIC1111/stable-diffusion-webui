#!/usr/bin/env python

import os
import sys
import logging
import torch
import diffusers
import safetensors
import safetensors.torch as sf

log = logging.getLogger("sdnext")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s | %(message)s')


def convert(model_id, output_name):
    if os.path.exists(output_name):
        log.error(f'Output already exists: {output_name}')
        return
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_id)
    metadata = { 'model_id': model_id }
    model = {}
    model['state_dict'] = vars(pipe)['_internal_dict']
    for k in model['state_dict'].keys():
        # print(k, getattr(pipe, k))
        model[k] = getattr(pipe, k)
    sf.save_model(model, output_name, metadata=metadata)
    # log.info(f'Saved model: {output_name}')

if __name__ == "__main__":
    sys.argv.pop(0)
    if len(sys.argv) < 2:
        log.info('Usage: hf-convert.py <model_id> <output_name>')
        sys.exit(1)
    log.debug(f'Packages: torch={torch.__version__} diffusers={diffusers.__version__} safetensors={safetensors.__version__}')
    convert(sys.argv[0], sys.argv[1])
