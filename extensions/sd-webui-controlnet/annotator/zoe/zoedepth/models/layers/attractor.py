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


@torch.jit.script
def exp_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a = attractor point, c = bin center, dc = shift in bin centermmary for exp_attractor

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor : Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return torch.exp(-alpha*(torch.abs(dx)**gamma)) * (dx)


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper. 

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1+alpha*dx.pow(gamma))


class AttractorLayer(nn.Module):
    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=1e-3, max_depth=10,
                 alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are bounded on the interval (min_depth, max_depth)
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors*2, 1, 1, 0),  # x2 for linear norm
            nn.ReLU(inplace=True)
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers normed and scaled; shape - n, nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        A = self._net(x)
        eps = 1e-3
        A = A + eps
        n, c, h, w = A.shape
        A = A.view(n, self.n_attractors, 2, h, w)
        A_normed = A / A.sum(dim=2, keepdim=True)  # n, a, 2, h, w
        A_normed = A[:, :, 0, ...]  # n, na, h, w

        b_prev = nn.functional.interpolate(
            b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev

        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor

        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(dist(A_normed.unsqueeze(
                2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                # .shape N, nbins, h, w
                delta_c += dist(A_normed[:, i, ...].unsqueeze(1) - b_centers)

            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = (self.max_depth - self.min_depth) * \
            b_new_centers + self.min_depth
        B_centers, _ = torch.sort(B_centers, dim=1)
        B_centers = torch.clip(B_centers, self.min_depth, self.max_depth)
        return b_new_centers, B_centers


class AttractorLayerUnnormed(nn.Module):
    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=1e-3, max_depth=10,
                 alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus()
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        A = self._net(x)
        n, c, h, w = A.shape

        b_prev = nn.functional.interpolate(
            b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev

        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor

        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(
                dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) -
                                b_centers)  # .shape N, nbins, h, w

            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers

        return b_new_centers, B_centers
