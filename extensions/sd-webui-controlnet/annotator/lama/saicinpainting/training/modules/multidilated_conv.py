import torch
import torch.nn as nn
import random
from annotator.lama.saicinpainting.training.modules.depthwise_sep_conv import DepthWiseSeperableConv

class MultidilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation_num=3, comb_mode='sum', equal_dim=True,
                 shared_weights=False, padding=1, min_dilation=1, shuffle_in_channels=False, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert(len(index) == out_dim)
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num

        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num

        conv_type = DepthWiseSeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(
                self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
            ))
            if i > 0 and shared_weights:
                convs[-1].weight = convs[0].weight
                convs[-1].bias = convs[0].bias
            dilation *= 2
        self.convs = nn.ModuleList(convs)

        self.shuffle_in_channels = shuffle_in_channels
        if self.shuffle_in_channels:
            # shuffle list as shuffling of tensors is nondeterministic
            in_channels_permute = list(range(in_dim))
            random.shuffle(in_channels_permute)
            # save as buffer so it is saved and loaded with checkpoint
            self.register_buffer('in_channels_permute', torch.tensor(in_channels_permute))

    def forward(self, x):
        if self.shuffle_in_channels:
            x = x[:, self.in_channels_permute]

        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start+dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out
