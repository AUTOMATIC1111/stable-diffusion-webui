# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import nn as nn
from torch.autograd import Function

from ..utils import ext_loader
from .ball_query import ball_query
from .knn import knn

ext_module = ext_loader.load_ext(
    '_ext', ['group_points_forward', 'group_points_backward'])


class QueryAndGroup(nn.Module):
    """Groups points with a ball query of radius.

    Args:
        max_radius (float): The maximum radius of the balls.
            If None is given, we will use kNN sampling instead of ball query.
        sample_num (int): Maximum number of features to gather in the ball.
        min_radius (float, optional): The minimum radius of the balls.
            Default: 0.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        return_grouped_xyz (bool, optional): Whether to return grouped xyz.
            Default: False.
        normalize_xyz (bool, optional): Whether to normalize xyz.
            Default: False.
        uniform_sample (bool, optional): Whether to sample uniformly.
            Default: False
        return_unique_cnt (bool, optional): Whether to return the count of
            unique samples. Default: False.
        return_grouped_idx (bool, optional): Whether to return grouped idx.
            Default: False.
    """

    def __init__(self,
                 max_radius,
                 sample_num,
                 min_radius=0,
                 use_xyz=True,
                 return_grouped_xyz=False,
                 normalize_xyz=False,
                 uniform_sample=False,
                 return_unique_cnt=False,
                 return_grouped_idx=False):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        self.return_grouped_idx = return_grouped_idx
        if self.return_unique_cnt:
            assert self.uniform_sample, \
                'uniform_sample should be True when ' \
                'returning the count of unique samples'
        if self.max_radius is None:
            assert not self.normalize_xyz, \
                'can not normalize grouped xyz when max_radius is None'

    def forward(self, points_xyz, center_xyz, features=None):
        """
        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) coordinates of the centriods.
            features (Tensor): (B, C, N) Descriptors of the features.

        Returns:
            Tensor: (B, 3 + C, npoint, sample_num) Grouped feature.
        """
        # if self.max_radius is None, we will perform kNN instead of ball query
        # idx is of shape [B, npoint, sample_num]
        if self.max_radius is None:
            idx = knn(self.sample_num, points_xyz, center_xyz, False)
            idx = idx.transpose(1, 2).contiguous()
        else:
            idx = ball_query(self.min_radius, self.max_radius, self.sample_num,
                             points_xyz, center_xyz)

        if self.uniform_sample:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0,
                        num_unique, (self.sample_num - num_unique, ),
                        dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        # (B, 3, npoint, sample_num)
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz_diff = grouped_xyz - \
            center_xyz.transpose(1, 2).unsqueeze(-1)  # relative offsets
        if self.normalize_xyz:
            grouped_xyz_diff /= self.max_radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                # (B, C + 3, npoint, sample_num)
                new_features = torch.cat([grouped_xyz_diff, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz
                    ), 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz_diff

        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_grouped_idx:
            ret.append(idx)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """Group xyz with feature.

    Args:
        use_xyz (bool): Whether to use xyz.
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            new_xyz (Tensor): new xyz coordinates of the features.
            features (Tensor): (B, C, N) features to group.

        Returns:
            Tensor: (B, C + 3, 1, N) Grouped feature.
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                # (B, 3 + C, 1, N)
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class GroupingOperation(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): (B, C, N) tensor of features to group.
            indices (Tensor): (B, npoint, nsample) the indices of
                features to group with.

        Returns:
            Tensor: (B, C, npoint, nsample) Grouped features.
        """
        features = features.contiguous()
        indices = indices.contiguous()

        B, nfeatures, nsample = indices.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        ext_module.group_points_forward(B, C, N, nfeatures, nsample, features,
                                        indices, output)

        ctx.for_backwards = (indices, N)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()

        grad_out_data = grad_out.data.contiguous()
        ext_module.group_points_backward(B, C, N, npoint, nsample,
                                         grad_out_data, idx,
                                         grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply
