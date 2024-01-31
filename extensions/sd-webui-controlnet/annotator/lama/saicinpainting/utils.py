import bisect
import functools
import logging
import numbers
import os
import signal
import sys
import traceback
import warnings

import torch
from pytorch_lightning import seed_everything

LOGGER = logging.getLogger(__name__)


def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}")


def sum_dict_with_prefix(target, cur_dict, prefix, default=0):
    for k, v in cur_dict.items():
        target_key = prefix + k
        target[target_key] = target.get(target_key, default) + v


def average_dicts(dict_list):
    result = {}
    norm = 1e-3
    for dct in dict_list:
        sum_dict_with_prefix(result, dct, '')
        norm += 1
    for k in list(result):
        result[k] /= norm
    return result


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def flatten_dict(dct):
    result = {}
    for k, v in dct.items():
        if isinstance(k, tuple):
            k = '_'.join(k)
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_dict(v).items():
                result[f'{k}_{sub_k}'] = sub_v
        else:
            result[k] = v
    return result


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class LadderRamp:
    def __init__(self, start_iters, values):
        self.start_iters = start_iters
        self.values = values
        assert len(values) == len(start_iters) + 1, (len(values), len(start_iters))

    def __call__(self, i):
        segment_i = bisect.bisect_right(self.start_iters, i)
        return self.values[segment_i]


def get_ramp(kind='ladder', **kwargs):
    if kind == 'linear':
        return LinearRamp(**kwargs)
    if kind == 'ladder':
        return LadderRamp(**kwargs)
    raise ValueError(f'Unexpected ramp kind: {kind}')


def print_traceback_handler(sig, frame):
    LOGGER.warning(f'Received signal {sig}')
    bt = ''.join(traceback.format_stack())
    LOGGER.warning(f'Requested stack trace:\n{bt}')


def register_debug_signal_handlers(sig=None, handler=print_traceback_handler):
    LOGGER.warning(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def handle_deterministic_config(config):
    seed = dict(config).get('seed', None)
    if seed is None:
        return False

    seed_everything(seed)
    return True


def get_shape(t):
    if torch.is_tensor(t):
        return tuple(t.shape)
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))


def get_has_ddp_rank():
    master_port = os.environ.get('MASTER_PORT', None)
    node_rank = os.environ.get('NODE_RANK', None)
    local_rank = os.environ.get('LOCAL_RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    has_rank = master_port is not None or node_rank is not None or local_rank is not None or world_size is not None
    return has_rank


def handle_ddp_subprocess():
    def main_decorator(main_func):
        @functools.wraps(main_func)
        def new_main(*args, **kwargs):
            # Trainer sets MASTER_PORT, NODE_RANK, LOCAL_RANK, WORLD_SIZE
            parent_cwd = os.environ.get('TRAINING_PARENT_WORK_DIR', None)
            has_parent = parent_cwd is not None
            has_rank = get_has_ddp_rank()
            assert has_parent == has_rank, f'Inconsistent state: has_parent={has_parent}, has_rank={has_rank}'

            if has_parent:
                # we are in the worker
                sys.argv.extend([
                    f'hydra.run.dir={parent_cwd}',
                    # 'hydra/hydra_logging=disabled',
                    # 'hydra/job_logging=disabled'
                ])
            # do nothing if this is a top-level process
            # TRAINING_PARENT_WORK_DIR is set in handle_ddp_parent_process after hydra initialization

            main_func(*args, **kwargs)
        return new_main
    return main_decorator


def handle_ddp_parent_process():
    parent_cwd = os.environ.get('TRAINING_PARENT_WORK_DIR', None)
    has_parent = parent_cwd is not None
    has_rank = get_has_ddp_rank()
    assert has_parent == has_rank, f'Inconsistent state: has_parent={has_parent}, has_rank={has_rank}'

    if parent_cwd is None:
        os.environ['TRAINING_PARENT_WORK_DIR'] = os.getcwd()

    return has_parent
