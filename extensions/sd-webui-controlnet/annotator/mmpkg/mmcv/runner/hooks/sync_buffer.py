# Copyright (c) OpenMMLab. All rights reserved.
from ..dist_utils import allreduce_params
from .hook import HOOKS, Hook


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch.

    Args:
        distributed (bool): Whether distributed training is used. It is
          effective only for distributed training. Defaults to True.
    """

    def __init__(self, distributed=True):
        self.distributed = distributed

    def after_epoch(self, runner):
        """All-reduce model buffers at the end of each epoch."""
        if self.distributed:
            allreduce_params(runner.model.buffers())
