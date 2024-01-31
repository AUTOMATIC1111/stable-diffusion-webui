# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['ball_query_forward'])


class BallQuery(Function):
    """Find nearby points in spherical space."""

    @staticmethod
    def forward(ctx, min_radius: float, max_radius: float, sample_num: int,
                xyz: torch.Tensor, center_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            min_radius (float): minimum radius of the balls.
            max_radius (float): maximum radius of the balls.
            sample_num (int): maximum number of features in the balls.
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) centers of the ball query.

        Returns:
            Tensor: (B, npoint, nsample) tensor with the indices of
                the features that form the query balls.
        """
        assert center_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert min_radius < max_radius

        B, N, _ = xyz.size()
        npoint = center_xyz.size(1)
        idx = xyz.new_zeros(B, npoint, sample_num, dtype=torch.int)

        ext_module.ball_query_forward(
            center_xyz,
            xyz,
            idx,
            b=B,
            n=N,
            m=npoint,
            min_radius=min_radius,
            max_radius=max_radius,
            nsample=sample_num)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply
