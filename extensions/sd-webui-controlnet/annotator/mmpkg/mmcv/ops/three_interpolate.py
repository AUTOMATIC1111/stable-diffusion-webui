from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['three_interpolate_forward', 'three_interpolate_backward'])


class ThreeInterpolate(Function):
    """Performs weighted linear interpolation on 3 features.

    Please refer to `Paper of PointNet++ <https://arxiv.org/abs/1706.02413>`_
    for more details.
    """

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert indices.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        ext_module.three_interpolate_forward(
            features, indices, weight, output, b=B, c=c, m=m, n=n)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = torch.cuda.FloatTensor(B, c, m).zero_()
        grad_out_data = grad_out.data.contiguous()

        ext_module.three_interpolate_backward(
            grad_out_data, idx, weight, grad_features.data, b=B, c=c, n=n, m=m)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
