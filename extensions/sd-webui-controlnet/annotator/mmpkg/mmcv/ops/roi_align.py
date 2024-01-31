# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import deprecated_api_warning, ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['roi_align_forward', 'roi_align_backward'])


class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
                 pool_mode, aligned):
        from ..onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        if has_custom_op:
            return g.op(
                'mmcv::MMCVRoiAlign',
                input,
                rois,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode,
                aligned_i=aligned)
        else:
            from torch.onnx.symbolic_opset9 import sub, squeeze
            from torch.onnx.symbolic_helper import _slice_helper
            from torch.onnx import TensorProtoDataType
            # batch_indices = rois[:, 0].long()
            batch_indices = _slice_helper(
                g, rois, axes=[1], starts=[0], ends=[1])
            batch_indices = squeeze(g, batch_indices, 1)
            batch_indices = g.op(
                'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
            # rois = rois[:, 1:]
            rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
            if aligned:
                # rois -= 0.5/spatial_scale
                aligned_offset = g.op(
                    'Constant',
                    value_t=torch.tensor([0.5 / spatial_scale],
                                         dtype=torch.float32))
                rois = sub(g, rois, aligned_offset)
            # roi align
            return g.op(
                'RoiAlign',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=max(0, sampling_ratio),
                mode_s=pool_mode)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        ext_module.roi_align_forward(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        ext_module.roi_align_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

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

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlign')
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.use_torchvision = use_torchvision

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            if 'aligned' in tv_roi_align.__code__.co_varnames:
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.aligned)
            else:
                if self.aligned:
                    rois -= rois.new_tensor([0.] +
                                            [0.5 / self.spatial_scale] * 4)
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        s += f'use_torchvision={self.use_torchvision})'
        return s
