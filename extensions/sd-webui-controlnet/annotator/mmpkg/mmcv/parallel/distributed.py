# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)

from annotator.mmpkg.mmcv import print_log
from annotator.mmpkg.mmcv.utils import TORCH_VERSION, digit_version
from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(DistributedDataParallel):
    """The DDP module that supports DataContainer.

    MMDDP has two main differences with PyTorch DDP:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data.
    - It implement two APIs ``train_step()`` and ``val_step()``.
    """

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """

        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output
