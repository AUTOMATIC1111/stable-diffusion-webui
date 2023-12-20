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


def log_binom(n, k, eps=1e-7):
    """ log(nCk) using stirling approximation """
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n-k) * torch.log(n-k+eps)


class LogBinomial(nn.Module):
    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes

        Args:
            n_classes (int, optional): number of output classes. Defaults to 256.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer('k_idx', torch.arange(
            0, n_classes).view(1, -1, 1, 1))
        self.register_buffer('K_minus_1', torch.Tensor(
            [self.K-1]).view(1, -1, 1, 1))

    def forward(self, x, t=1., eps=1e-4):
        """Compute log binomial distribution for x

        Args:
            x (torch.Tensor - NCHW): probabilities
            t (float, torch.Tensor - NCHW, optional): Temperature of distribution. Defaults to 1..
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.

        Returns:
            torch.Tensor -NCHW: log binomial distribution logbinomial(p;t)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # make it nchw

        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = log_binom(self.K_minus_1, self.k_idx) + self.k_idx * \
            torch.log(x) + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        return self.act(y/t, dim=1)


class ConditionalLogBinomial(nn.Module):
    def __init__(self, in_features, condition_dim, n_classes=256, bottleneck_factor=2, p_eps=1e-4, max_temp=50, min_temp=1e-7, act=torch.softmax):
        """Conditional Log Binomial distribution

        Args:
            in_features (int): number of input channels in main feature
            condition_dim (int): number of input channels in condition feature
            n_classes (int, optional): Number of classes. Defaults to 256.
            bottleneck_factor (int, optional): Hidden dim factor. Defaults to 2.
            p_eps (float, optional): small eps value. Defaults to 1e-4.
            max_temp (float, optional): Maximum temperature of output distribution. Defaults to 50.
            min_temp (float, optional): Minimum temperature of output distribution. Defaults to 1e-7.
        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck,
                      kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            # 2 for p linear norm, 2 for t linear norm
            nn.Conv2d(bottleneck, 2+2, kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )

    def forward(self, x, cond):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Main feature
            cond (torch.Tensor - NCHW): condition feature

        Returns:
            torch.Tensor: Output log binomial distribution
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]

        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])

        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp

        return self.log_binomial_transform(p, t)
