import torch


def make_weight_cp(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)


def rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)
