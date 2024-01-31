# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from annotator.mmpkg.mmcv.cnn import NORM_LAYERS
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'sync_bn_forward_mean', 'sync_bn_forward_var', 'sync_bn_forward_output',
    'sync_bn_backward_param', 'sync_bn_backward_data'
])


class SyncBatchNormFunction(Function):

    @staticmethod
    def symbolic(g, input, running_mean, running_var, weight, bias, momentum,
                 eps, group, group_size, stats_mode):
        return g.op(
            'mmcv::MMCVSyncBatchNorm',
            input,
            running_mean,
            running_var,
            weight,
            bias,
            momentum_f=momentum,
            eps_f=eps,
            group_i=group,
            group_size_i=group_size,
            stats_mode=stats_mode)

    @staticmethod
    def forward(self, input, running_mean, running_var, weight, bias, momentum,
                eps, group, group_size, stats_mode):
        self.momentum = momentum
        self.eps = eps
        self.group = group
        self.group_size = group_size
        self.stats_mode = stats_mode

        assert isinstance(
                   input, (torch.HalfTensor, torch.FloatTensor,
                           torch.cuda.HalfTensor, torch.cuda.FloatTensor)), \
               f'only support Half or Float Tensor, but {input.type()}'
        output = torch.zeros_like(input)
        input3d = input.flatten(start_dim=2)
        output3d = output.view_as(input3d)
        num_channels = input3d.size(1)

        # ensure mean/var/norm/std are initialized as zeros
        # ``torch.empty()`` does not guarantee that
        mean = torch.zeros(
            num_channels, dtype=torch.float, device=input3d.device)
        var = torch.zeros(
            num_channels, dtype=torch.float, device=input3d.device)
        norm = torch.zeros_like(
            input3d, dtype=torch.float, device=input3d.device)
        std = torch.zeros(
            num_channels, dtype=torch.float, device=input3d.device)

        batch_size = input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_forward_mean(input3d, mean)
            batch_flag = torch.ones([1], device=mean.device, dtype=mean.dtype)
        else:
            # skip updating mean and leave it as zeros when the input is empty
            batch_flag = torch.zeros([1], device=mean.device, dtype=mean.dtype)

        # synchronize mean and the batch flag
        vec = torch.cat([mean, batch_flag])
        if self.stats_mode == 'N':
            vec *= batch_size
        if self.group_size > 1:
            dist.all_reduce(vec, group=self.group)
        total_batch = vec[-1].detach()
        mean = vec[:num_channels]

        if self.stats_mode == 'default':
            mean = mean / self.group_size
        elif self.stats_mode == 'N':
            mean = mean / total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        # leave var as zeros when the input is empty
        if batch_size > 0:
            ext_module.sync_bn_forward_var(input3d, mean, var)

        if self.stats_mode == 'N':
            var *= batch_size
        if self.group_size > 1:
            dist.all_reduce(var, group=self.group)

        if self.stats_mode == 'default':
            var /= self.group_size
        elif self.stats_mode == 'N':
            var /= total_batch.clamp(min=1)
        else:
            raise NotImplementedError

        # if the total batch size over all the ranks is zero,
        # we should not update the statistics in the current batch
        update_flag = total_batch.clamp(max=1)
        momentum = update_flag * self.momentum
        ext_module.sync_bn_forward_output(
            input3d,
            mean,
            var,
            weight,
            bias,
            running_mean,
            running_var,
            norm,
            std,
            output3d,
            eps=self.eps,
            momentum=momentum,
            group_size=self.group_size)
        self.save_for_backward(norm, std, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(self, grad_output):
        norm, std, weight = self.saved_tensors
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        grad_input = torch.zeros_like(grad_output)
        grad_output3d = grad_output.flatten(start_dim=2)
        grad_input3d = grad_input.view_as(grad_output3d)

        batch_size = grad_input3d.size(0)
        if batch_size > 0:
            ext_module.sync_bn_backward_param(grad_output3d, norm, grad_weight,
                                              grad_bias)

        # all reduce
        if self.group_size > 1:
            dist.all_reduce(grad_weight, group=self.group)
            dist.all_reduce(grad_bias, group=self.group)
            grad_weight /= self.group_size
            grad_bias /= self.group_size

        if batch_size > 0:
            ext_module.sync_bn_backward_data(grad_output3d, weight,
                                             grad_weight, grad_bias, norm, std,
                                             grad_input3d)

        return grad_input, None, None, grad_weight, grad_bias, \
            None, None, None, None, None


@NORM_LAYERS.register_module(name='MMSyncBN')
class SyncBatchNorm(Module):
    """Synchronized Batch Normalization.

    Args:
        num_features (int): number of features/chennels in input tensor
        eps (float, optional): a value added to the denominator for numerical
            stability. Defaults to 1e-5.
        momentum (float, optional): the value used for the running_mean and
            running_var computation. Defaults to 0.1.
        affine (bool, optional): whether to use learnable affine parameters.
            Defaults to True.
        track_running_stats (bool, optional): whether to track the running
            mean and variance during training. When set to False, this
            module does not track such statistics, and initializes statistics
            buffers ``running_mean`` and ``running_var`` as ``None``. When
            these buffers are ``None``, this module always uses batch
            statistics in both training and eval modes. Defaults to True.
        group (int, optional): synchronization of stats happen within
            each process group individually. By default it is synchronization
            across the whole world. Defaults to None.
        stats_mode (str, optional): The statistical mode. Available options
            includes ``'default'`` and ``'N'``. Defaults to 'default'.
            When ``stats_mode=='default'``, it computes the overall statistics
            using those from each worker with equal weight, i.e., the
            statistics are synchronized and simply divied by ``group``. This
            mode will produce inaccurate statistics when empty tensors occur.
            When ``stats_mode=='N'``, it compute the overall statistics using
            the total number of batches in each worker ignoring the number of
            group, i.e., the statistics are synchronized and then divied by
            the total batch ``N``. This mode is beneficial when empty tensors
            occur during training, as it average the total mean by the real
            number of batch.
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 group=None,
                 stats_mode='default'):
        super(SyncBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        group = dist.group.WORLD if group is None else group
        self.group = group
        self.group_size = dist.get_world_size(group)
        assert stats_mode in ['default', 'N'], \
            f'"stats_mode" only accepts "default" and "N", got "{stats_mode}"'
        self.stats_mode = stats_mode
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()  # pytorch use ones_()
            self.bias.data.zero_()

    def forward(self, input):
        if input.dim() < 2:
            raise ValueError(
                f'expected at least 2D input, got {input.dim()}D input')
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats:
            return SyncBatchNormFunction.apply(
                input, self.running_mean, self.running_var, self.weight,
                self.bias, exponential_average_factor, self.eps, self.group,
                self.group_size, self.stats_mode)
        else:
            return F.batch_norm(input, self.running_mean, self.running_var,
                                self.weight, self.bias, False,
                                exponential_average_factor, self.eps)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'({self.num_features}, '
        s += f'eps={self.eps}, '
        s += f'momentum={self.momentum}, '
        s += f'affine={self.affine}, '
        s += f'track_running_stats={self.track_running_stats}, '
        s += f'group_size={self.group_size},'
        s += f'stats_mode={self.stats_mode})'
        return s
