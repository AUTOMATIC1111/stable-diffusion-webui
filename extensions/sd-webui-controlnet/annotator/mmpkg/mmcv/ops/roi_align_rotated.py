# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['roi_align_rotated_forward', 'roi_align_rotated_backward'])


class RoIAlignRotatedFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, out_size, spatial_scale, sample_num,
                 aligned, clockwise):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        return g.op(
            'mmcv::MMCVRoIAlignRotated',
            features,
            rois,
            output_height_i=out_h,
            output_width_i=out_h,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sample_num,
            aligned_i=aligned,
            clockwise_i=clockwise)

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                sample_num=0,
                aligned=True,
                clockwise=False):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.aligned = aligned
        ctx.clockwise = clockwise
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        ext_module.roi_align_rotated_forward(
            features,
            rois,
            output,
            pooled_height=out_h,
            pooled_width=out_w,
            spatial_scale=spatial_scale,
            sample_num=sample_num,
            aligned=aligned,
            clockwise=clockwise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        aligned = ctx.aligned
        clockwise = ctx.clockwise
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size

        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            ext_module.roi_align_rotated_backward(
                grad_output.contiguous(),
                rois,
                grad_input,
                pooled_height=out_h,
                pooled_width=out_w,
                spatial_scale=spatial_scale,
                sample_num=sample_num,
                aligned=aligned,
                clockwise=clockwise)
        return grad_input, grad_rois, None, None, None, None, None


roi_align_rotated = RoIAlignRotatedFunction.apply


class RoIAlignRotated(nn.Module):
    """RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    Args:
        out_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sample_num (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=True,
                 clockwise=False):
        super(RoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.aligned = aligned
        self.clockwise = clockwise

    def forward(self, features, rois):
        return RoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                             self.spatial_scale,
                                             self.sample_num, self.aligned,
                                             self.clockwise)
