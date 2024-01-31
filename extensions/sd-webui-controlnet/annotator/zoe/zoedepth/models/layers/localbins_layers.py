# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn


class SeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are bounded on (min_depth, max_depth) interval.

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Min depth value. Defaults to 1e-3.
            max_depth (float, optional): Max depth value. Defaults to 10.
        """
        super().__init__()
        self.version = "1_1"
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B = self._net(x)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * \
            B_widths_normed  # .shape NCHW
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(
            B_widths, (0, 0, 0, 0, 1, 0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:, 1:, ...])
        return B_widths_normed, B_centers


class SeedBinRegressorUnnormed(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = "1_1"
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.Softplus()
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


class Projector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP

        Args:
            in_features (int): input channels
            out_features (int): output channels
            mlp_dim (int, optional): hidden dimension. Defaults to 128.
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)



class LinearSplitter(nn.Module):
    def __init__(self, in_features, prev_nbins, split_factor=2, mlp_dim=128, min_depth=1e-3, max_depth=10):
        super().__init__()

        self.prev_nbins = prev_nbins
        self.split_factor = split_factor
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mlp_dim, prev_nbins * split_factor, 1, 1, 0),
            nn.ReLU()
        )
    
    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        x : feature block; shape - n, c, h, w
        b_prev : previous bin widths normed; shape - n, prev_nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding
        S = self._net(x)
        eps = 1e-3
        S = S + eps
        n, c, h, w = S.shape
        S = S.view(n, self.prev_nbins, self.split_factor, h, w)
        S_normed = S / S.sum(dim=2, keepdim=True)  # fractional splits

        b_prev = nn.functional.interpolate(b_prev, (h,w), mode='bilinear', align_corners=True)
        

        b_prev = b_prev / b_prev.sum(dim=1, keepdim=True)  # renormalize for gurantees
        # print(b_prev.shape, S_normed.shape)
        # if is_for_query:(1).expand(-1, b_prev.size(0)//n, -1, -1, -1, -1).flatten(0,1)  # TODO ? can replace all this with a single torch.repeat?
        b = b_prev.unsqueeze(2) * S_normed
        b = b.flatten(1,2)  # .shape n, prev_nbins * split_factor, h, w

        # calculate bin centers for loss calculation
        B_widths = (self.max_depth - self.min_depth) * b  # .shape N, nprev * splitfactor, H, W
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0,0,0,0,1,0), mode='constant', value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:,1:,...])
        return b, B_centers