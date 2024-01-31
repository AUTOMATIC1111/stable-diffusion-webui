# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)

from annotator.mmpkg.mmcv.utils import TORCH_VERSION, digit_version
from .registry import MODULE_WRAPPERS
from .scatter_gather import scatter_kwargs


@MODULE_WRAPPERS.register_module()
class MMDistributedDataParallel(nn.Module):

    def __init__(self,
                 module,
                 dim=0,
                 broadcast_buffers=True,
                 bucket_cap_mb=25):
        super(MMDistributedDataParallel, self).__init__()
        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers

        self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
        self._sync_params()

    def _dist_broadcast_coalesced(self, tensors, buffer_size):
        for tensors in _take_tensors(tensors, buffer_size):
            flat_tensors = _flatten_dense_tensors(tensors)
            dist.broadcast(flat_tensors, 0)
            for tensor, synced in zip(
                    tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
                tensor.copy_(synced)

    def _sync_params(self):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._dist_broadcast_coalesced(module_states,
                                           self.broadcast_bucket_size)
        if self.broadcast_buffers:
            if (TORCH_VERSION != 'parrots'
                    and digit_version(TORCH_VERSION) < digit_version('1.0')):
                buffers = [b.data for b in self.module._all_buffers()]
            else:
                buffers = [b.data for b in self.module.buffers()]
            if len(buffers) > 0:
                self._dist_broadcast_coalesced(buffers,
                                               self.broadcast_bucket_size)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        return self.module(*inputs[0], **kwargs[0])

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.train_step(*inputs[0], **kwargs[0])
        return output

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs,
                                      [torch.cuda.current_device()])
        output = self.module.val_step(*inputs[0], **kwargs[0])
        return output
