import glob
import os
from modules import sd_models, shared
import torch
from safetensors.torch import load_file

model_name = None
backup_weights = {}

def list_pluggables(path):
    res = {}
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.safetensors'), recursive=True)):
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f"({sd_models.model_hash(filename)})"] = filename
    return res

def load_pluggable(filename):
    global model_name, backup_weights
    if filename == 'None':
        restore_weights()
        return
    print('loading:', filename)
    path = shared.pluggables[filename]
    if model_name != shared.opts.sd_model_checkpoint:
        backup_weights = {}
        model_name = shared.opts.sd_model_checkpoint
    pluggable = load_file(path, device='cuda')
    for k, v in shared.sd_model.model.named_parameters():
        if k in pluggable:
            if k not in backup_weights:
                backup_weights[k] = v.clone().detach()
            with torch.no_grad():
                v[:] = pluggable[k]
            print(f'plugged in from {filename} {k}')
        elif k in backup_weights:
            with torch.no_grad():
                v[:] = backup_weights[k]
            del backup_weights[k]
            print(f'restored {k}')


def restore_weights():
    global backup_weights
    if model_name == shared.opts.sd_model_checkpoint:
        for k, v in shared.sd_model.model.named_parameters():
            if k in backup_weights:
                with torch.no_grad():
                    v[:] = backup_weights[k]
                print(f'restored {k}')
    backup_weights = {}
