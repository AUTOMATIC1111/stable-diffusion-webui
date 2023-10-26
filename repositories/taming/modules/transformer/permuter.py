import torch
import torch.nn as nn
import numpy as np


class AbstractPermuter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, reverse=False):
        raise NotImplementedError


class Identity(AbstractPermuter):
    def __init__(self):
        super().__init__()

    def forward(self, x, reverse=False):
        return x


class Subsample(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        C = 1
        indices = np.arange(H*W).reshape(C,H,W)
        while min(H, W) > 1:
            indices = indices.reshape(C,H//2,2,W//2,2)
            indices = indices.transpose(0,2,4,1,3)
            indices = indices.reshape(C*4,H//2, W//2)
            H = H//2
            W = W//2
            C = C*4
        assert H == W == 1
        idx = torch.tensor(indices.ravel())
        self.register_buffer('forward_shuffle_idx',
                             nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx',
                             nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


def mortonify(i, j):
    """(i,j) index to linear morton code"""
    i = np.uint64(i)
    j = np.uint64(j)

    z = np.uint(0)

    for pos in range(32):
        z = (z |
             ((j & (np.uint64(1) << np.uint64(pos))) << np.uint64(pos)) |
             ((i & (np.uint64(1) << np.uint64(pos))) << np.uint64(pos+1))
             )
    return z


class ZCurve(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        reverseidx = [np.int64(mortonify(i,j)) for i in range(H) for j in range(W)]
        idx = np.argsort(reverseidx)
        idx = torch.tensor(idx)
        reverseidx = torch.tensor(reverseidx)
        self.register_buffer('forward_shuffle_idx',
                             idx)
        self.register_buffer('backward_shuffle_idx',
                             reverseidx)

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class SpiralOut(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        assert H == W
        size = W
        indices = np.arange(size*size).reshape(size,size)

        i0 = size//2
        j0 = size//2-1

        i = i0
        j = j0

        idx = [indices[i0, j0]]
        step_mult = 0
        for c in range(1, size//2+1):
            step_mult += 1
            # steps left
            for k in range(step_mult):
                i = i - 1
                j = j
                idx.append(indices[i, j])

            # step down
            for k in range(step_mult):
                i = i
                j = j + 1
                idx.append(indices[i, j])

            step_mult += 1
            if c < size//2:
                # step right
                for k in range(step_mult):
                    i = i + 1
                    j = j
                    idx.append(indices[i, j])

                # step up
                for k in range(step_mult):
                    i = i
                    j = j - 1
                    idx.append(indices[i, j])
            else:
                # end reached
                for k in range(step_mult-1):
                    i = i + 1
                    idx.append(indices[i, j])

        assert len(idx) == size*size
        idx = torch.tensor(idx)
        self.register_buffer('forward_shuffle_idx', idx)
        self.register_buffer('backward_shuffle_idx', torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class SpiralIn(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        assert H == W
        size = W
        indices = np.arange(size*size).reshape(size,size)

        i0 = size//2
        j0 = size//2-1

        i = i0
        j = j0

        idx = [indices[i0, j0]]
        step_mult = 0
        for c in range(1, size//2+1):
            step_mult += 1
            # steps left
            for k in range(step_mult):
                i = i - 1
                j = j
                idx.append(indices[i, j])

            # step down
            for k in range(step_mult):
                i = i
                j = j + 1
                idx.append(indices[i, j])

            step_mult += 1
            if c < size//2:
                # step right
                for k in range(step_mult):
                    i = i + 1
                    j = j
                    idx.append(indices[i, j])

                # step up
                for k in range(step_mult):
                    i = i
                    j = j - 1
                    idx.append(indices[i, j])
            else:
                # end reached
                for k in range(step_mult-1):
                    i = i + 1
                    idx.append(indices[i, j])

        assert len(idx) == size*size
        idx = idx[::-1]
        idx = torch.tensor(idx)
        self.register_buffer('forward_shuffle_idx', idx)
        self.register_buffer('backward_shuffle_idx', torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class Random(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        indices = np.random.RandomState(1).permutation(H*W)
        idx = torch.tensor(indices.ravel())
        self.register_buffer('forward_shuffle_idx', idx)
        self.register_buffer('backward_shuffle_idx', torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


class AlternateParsing(AbstractPermuter):
    def __init__(self, H, W):
        super().__init__()
        indices = np.arange(W*H).reshape(H,W)
        for i in range(1, H, 2):
            indices[i, :] = indices[i, ::-1]
        idx = indices.flatten()
        assert len(idx) == H*W
        idx = torch.tensor(idx)
        self.register_buffer('forward_shuffle_idx', idx)
        self.register_buffer('backward_shuffle_idx', torch.argsort(idx))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx]
        else:
            return x[:, self.backward_shuffle_idx]


if __name__ == "__main__":
    p0 = AlternateParsing(16, 16)
    print(p0.forward_shuffle_idx)
    print(p0.backward_shuffle_idx)

    x = torch.randint(0, 768, size=(11, 256))
    y = p0(x)
    xre = p0(y, reverse=True)
    assert torch.equal(x, xre)

    p1 = SpiralOut(2, 2)
    print(p1.forward_shuffle_idx)
    print(p1.backward_shuffle_idx)
