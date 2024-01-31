# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.autograd import Function

import annotator.mmpkg.mmcv as mmcv
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['roiaware_pool3d_forward', 'roiaware_pool3d_backward'])


class RoIAwarePool3d(nn.Module):
    """Encode the geometry-specific features of each 3D proposal.

    Please refer to `PartA2 <https://arxiv.org/pdf/1907.03670.pdf>`_ for more
    details.

    Args:
        out_size (int or tuple): The size of output features. n or
            [n1, n2, n3].
        max_pts_per_voxel (int, optional): The maximum number of points per
            voxel. Default: 128.
        mode (str, optional): Pooling method of RoIAware, 'max' or 'avg'.
            Default: 'max'.
    """

    def __init__(self, out_size, max_pts_per_voxel=128, mode='max'):
        super().__init__()

        self.out_size = out_size
        self.max_pts_per_voxel = max_pts_per_voxel
        assert mode in ['max', 'avg']
        pool_mapping = {'max': 0, 'avg': 1}
        self.mode = pool_mapping[mode]

    def forward(self, rois, pts, pts_feature):
        """
        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois.
            pts (torch.Tensor): [npoints, 3], coordinates of input points.
            pts_feature (torch.Tensor): [npoints, C], features of input points.

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C]
        """

        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature,
                                            self.out_size,
                                            self.max_pts_per_voxel, self.mode)


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_per_voxel,
                mode):
        """
        Args:
            rois (torch.Tensor): [N, 7], in LiDAR coordinate,
                (x, y, z) is the bottom center of rois.
            pts (torch.Tensor): [npoints, 3], coordinates of input points.
            pts_feature (torch.Tensor): [npoints, C], features of input points.
            out_size (int or tuple): The size of output features. n or
                [n1, n2, n3].
            max_pts_per_voxel (int): The maximum number of points per voxel.
                Default: 128.
            mode (int): Pooling method of RoIAware, 0 (max pool) or 1 (average
                pool).

        Returns:
            pooled_features (torch.Tensor): [N, out_x, out_y, out_z, C], output
                pooled features.
        """

        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            assert mmcv.is_tuple_of(out_size, int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros(
            (num_rois, out_x, out_y, out_z, max_pts_per_voxel),
            dtype=torch.int)

        ext_module.roiaware_pool3d_forward(rois, pts, pts_feature, argmax,
                                           pts_idx_of_voxels, pooled_features,
                                           mode)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, mode,
                                            num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        ret = ctx.roiaware_pool3d_for_backward
        pts_idx_of_voxels, argmax, mode, num_pts, num_channels = ret

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        ext_module.roiaware_pool3d_backward(pts_idx_of_voxels, argmax,
                                            grad_out.contiguous(), grad_in,
                                            mode)

        return None, None, grad_in, None, None, None
