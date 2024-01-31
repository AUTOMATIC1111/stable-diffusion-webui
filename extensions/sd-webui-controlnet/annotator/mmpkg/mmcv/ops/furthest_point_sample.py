import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'furthest_point_sampling_forward',
    'furthest_point_sampling_with_dist_forward'
])


class FurthestPointSampling(Function):
    """Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance."""

    @staticmethod
    def forward(ctx, points_xyz: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """
        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_xyz.is_contiguous()

        B, N = points_xyz.size()[:2]
        output = torch.cuda.IntTensor(B, num_points)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        ext_module.furthest_point_sampling_forward(
            points_xyz,
            temp,
            output,
            b=B,
            n=N,
            m=num_points,
        )
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(Function):
    """Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance."""

    @staticmethod
    def forward(ctx, points_dist: torch.Tensor,
                num_points: int) -> torch.Tensor:
        """
        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        assert points_dist.is_contiguous()

        B, N, _ = points_dist.size()
        output = points_dist.new_zeros([B, num_points], dtype=torch.int32)
        temp = points_dist.new_zeros([B, N]).fill_(1e10)

        ext_module.furthest_point_sampling_with_dist_forward(
            points_dist, temp, output, b=B, n=N, m=num_points)
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
