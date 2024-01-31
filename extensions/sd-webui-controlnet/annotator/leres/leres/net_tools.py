import importlib
import torch
import os
from collections import OrderedDict


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'annotator.leres.leres.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to f1ind function: %s', func_name)
        raise

def load_ckpt(args, depth_model, shift_model, focal_model):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        print("loading checkpoint %s" % args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt)
        if shift_model is not None:
            shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                    strict=True)
        if focal_model is not None:
            focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                    strict=True)
        depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                    strict=True)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict