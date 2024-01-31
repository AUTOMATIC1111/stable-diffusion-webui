import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['knn_forward'])


class KNN(Function):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    def forward(ctx,
                k: int,
                xyz: torch.Tensor,
                center_xyz: torch.Tensor = None,
                transposed: bool = False) -> torch.Tensor:
        """
        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor, optional): (B, npoint, 3) if transposed ==
                False, else (B, 3, npoint). centers of the knn query.
                Default: None.
            transposed (bool, optional): whether the input tensors are
                transposed. Should not explicitly use this keyword when
                calling knn (=KNN.apply), just add the fourth param.
                Default: False.

        Returns:
            Tensor: (B, k, npoint) tensor with the indices of
                the features that form k-nearest neighbours.
        """
        assert (k > 0) & (k < 100), 'k should be in range(0, 100)'

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        assert xyz.is_contiguous()  # [B, N, 3]
        assert center_xyz.is_contiguous()  # [B, npoint, 3]

        center_xyz_device = center_xyz.get_device()
        assert center_xyz_device == xyz.get_device(), \
            'center_xyz and xyz should be put on the same device'
        if torch.cuda.current_device() != center_xyz_device:
            torch.cuda.set_device(center_xyz_device)

        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]

        idx = center_xyz.new_zeros((B, npoint, k)).int()
        dist2 = center_xyz.new_zeros((B, npoint, k)).float()

        ext_module.knn_forward(
            xyz, center_xyz, idx, dist2, b=B, n=N, m=npoint, nsample=k)
        # idx shape to [B, k, npoint]
        idx = idx.transpose(2, 1).contiguous()
        if torch.__version__ != 'parrots':
            ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply
