# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['deform_roi_pool_forward', 'deform_roi_pool_backward'])


class DeformRoIPoolFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, offset, output_size, spatial_scale,
                 sampling_ratio, gamma):
        return g.op(
            'mmcv::MMCVDeformRoIPool',
            input,
            rois,
            offset,
            pooled_height_i=output_size[0],
            pooled_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_f=sampling_ratio,
            gamma_f=gamma)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                offset,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                gamma=0.1):
        if offset is None:
            offset = input.new_zeros(0)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = float(spatial_scale)
        ctx.sampling_ratio = int(sampling_ratio)
        ctx.gamma = float(gamma)

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)

        ext_module.deform_roi_pool_forward(
            input,
            rois,
            offset,
            output,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma)

        ctx.save_for_backward(input, rois, offset)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, offset = ctx.saved_tensors
        grad_input = grad_output.new_zeros(input.shape)
        grad_offset = grad_output.new_zeros(offset.shape)

        ext_module.deform_roi_pool_backward(
            grad_output,
            input,
            rois,
            offset,
            grad_input,
            grad_offset,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            gamma=ctx.gamma)
        if grad_offset.numel() == 0:
            grad_offset = None
        return grad_input, None, grad_offset, None, None, None, None


deform_roi_pool = DeformRoIPoolFunction.apply


class DeformRoIPool(nn.Module):

    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(DeformRoIPool, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.gamma = float(gamma)

    def forward(self, input, rois, offset=None):
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)


class DeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(DeformRoIPoolPack, self).__init__(output_size, spatial_scale,
                                                sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        return deform_roi_pool(input, rois, offset, self.output_size,
                               self.spatial_scale, self.sampling_ratio,
                               self.gamma)


class ModulatedDeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size,
                 output_channels,
                 deform_fc_channels=1024,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 gamma=0.1):
        super(ModulatedDeformRoIPoolPack,
              self).__init__(output_size, spatial_scale, sampling_ratio, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

        self.mask_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 1),
            nn.Sigmoid())
        self.mask_fc[2].weight.data.zero_()
        self.mask_fc[2].bias.data.zero_()

    def forward(self, input, rois):
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        mask = self.mask_fc(x.view(rois_num, -1))
        mask = mask.view(rois_num, 1, self.output_size[0], self.output_size[1])
        d = deform_roi_pool(input, rois, offset, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.gamma)
        return d * mask
