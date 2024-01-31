from torch import nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['roipoint_pool3d_forward'])


class RoIPointPool3d(nn.Module):
    """Encode the geometry-specific features of each 3D proposal.

    Please refer to `Paper of PartA2 <https://arxiv.org/pdf/1907.03670.pdf>`_
    for more details.

    Args:
        num_sampled_points (int, optional): Number of samples in each roi.
            Default: 512.
    """

    def __init__(self, num_sampled_points=512):
        super().__init__()
        self.num_sampled_points = num_sampled_points

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, C).
            point_features (torch.Tensor): Features of input points whose shape
                is (B, N, C).
            boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).

        Returns:
            pooled_features (torch.Tensor): The output pooled features whose
                shape is (B, M, 512, 3 + C).
            pooled_empty_flag (torch.Tensor): Empty flag whose shape is (B, M).
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d,
                                            self.num_sampled_points)


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx, points, point_features, boxes3d, num_sampled_points=512):
        """
        Args:
            points (torch.Tensor): Input points whose shape is (B, N, C).
            point_features (torch.Tensor): Features of input points whose shape
                is (B, N, C).
            boxes3d (B, M, 7), Input bounding boxes whose shape is (B, M, 7).
            num_sampled_points (int, optional): The num of sampled points.
                Default: 512.

        Returns:
            pooled_features (torch.Tensor): The output pooled features whose
                shape is (B, M, 512, 3 + C).
            pooled_empty_flag (torch.Tensor): Empty flag whose shape is (B, M).
        """
        assert len(points.shape) == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[
            1], point_features.shape[2]
        pooled_boxes3d = boxes3d.view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros(
            (batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros(
            (batch_size, boxes_num)).int()

        ext_module.roipoint_pool3d_forward(points.contiguous(),
                                           pooled_boxes3d.contiguous(),
                                           point_features.contiguous(),
                                           pooled_features, pooled_empty_flag)

        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
