from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from annotator.lama.saicinpainting.training.modules.base import get_conv_block_ctor, get_activation
from annotator.lama.saicinpainting.training.modules.pix2pixhd import ResnetBlock


class ResNetHead(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default', activation=nn.ReLU(True)):
        assert (n_blocks >= 0)
        super(ResNetHead, self).__init__()

        conv_layer = get_conv_block_ctor(conv_kind)

        model = [nn.ReflectionPad2d(3),
                 conv_layer(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      activation]

        mult = 2 ** n_downsampling

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                  conv_kind=conv_kind)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResNetTail(nn.Module):
    def __init__(self, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), add_out_act=False, out_extra_layers_n=0,
                 add_in_proj=None):
        assert (n_blocks >= 0)
        super(ResNetTail, self).__init__()

        mult = 2 ** n_downsampling

        model = []

        if add_in_proj is not None:
            model.append(nn.Conv2d(add_in_proj, ngf * mult, kernel_size=1))

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                  conv_kind=conv_kind)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      up_norm_layer(int(ngf * mult / 2)),
                      up_activation]
        self.model = nn.Sequential(*model)

        out_layers = []
        for _ in range(out_extra_layers_n):
            out_layers += [nn.Conv2d(ngf, ngf, kernel_size=1, padding=0),
                           up_norm_layer(ngf),
                           up_activation]
        out_layers += [nn.ReflectionPad2d(3),
                       nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        if add_out_act:
            out_layers.append(get_activation('tanh' if add_out_act is True else add_out_act))

        self.out_proj = nn.Sequential(*out_layers)

    def forward(self, input, return_last_act=False):
        features = self.model(input)
        out = self.out_proj(features)
        if return_last_act:
            return out, features
        else:
            return out


class MultiscaleResNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks_head=2, n_blocks_tail=6, n_scales=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', conv_kind='default', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True), add_out_act=False, out_extra_layers_n=0,
                 out_cumulative=False, return_only_hr=False):
        super().__init__()

        self.heads = nn.ModuleList([ResNetHead(input_nc, ngf=ngf, n_downsampling=n_downsampling,
                                               n_blocks=n_blocks_head, norm_layer=norm_layer, padding_type=padding_type,
                                               conv_kind=conv_kind, activation=activation)
                                    for i in range(n_scales)])
        tail_in_feats = ngf * (2 ** n_downsampling) + ngf
        self.tails = nn.ModuleList([ResNetTail(output_nc,
                                               ngf=ngf, n_downsampling=n_downsampling,
                                               n_blocks=n_blocks_tail, norm_layer=norm_layer, padding_type=padding_type,
                                               conv_kind=conv_kind, activation=activation, up_norm_layer=up_norm_layer,
                                               up_activation=up_activation, add_out_act=add_out_act,
                                               out_extra_layers_n=out_extra_layers_n,
                                               add_in_proj=None if (i == n_scales - 1) else tail_in_feats)
                                    for i in range(n_scales)])

        self.out_cumulative = out_cumulative
        self.return_only_hr = return_only_hr

    @property
    def num_scales(self):
        return len(self.heads)

    def forward(self, ms_inputs: List[torch.Tensor], smallest_scales_num: Optional[int] = None) \
        -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        :param ms_inputs: List of inputs of different resolutions from HR to LR
        :param smallest_scales_num: int or None, number of smallest scales to take at input
        :return: Depending on return_only_hr:
            True: Only the most HR output
            False: List of outputs of different resolutions from HR to LR
        """
        if smallest_scales_num is None:
            assert len(self.heads) == len(ms_inputs), (len(self.heads), len(ms_inputs), smallest_scales_num)
            smallest_scales_num = len(self.heads)
        else:
            assert smallest_scales_num == len(ms_inputs) <= len(self.heads), (len(self.heads), len(ms_inputs), smallest_scales_num)

        cur_heads = self.heads[-smallest_scales_num:]
        ms_features = [cur_head(cur_inp) for cur_head, cur_inp in zip(cur_heads, ms_inputs)]

        all_outputs = []
        prev_tail_features = None
        for i in range(len(ms_features)):
            scale_i = -i - 1

            cur_tail_input = ms_features[-i - 1]
            if prev_tail_features is not None:
                if prev_tail_features.shape != cur_tail_input.shape:
                    prev_tail_features = F.interpolate(prev_tail_features, size=cur_tail_input.shape[2:],
                                                       mode='bilinear', align_corners=False)
                cur_tail_input = torch.cat((cur_tail_input, prev_tail_features), dim=1)

            cur_out, cur_tail_feats = self.tails[scale_i](cur_tail_input, return_last_act=True)

            prev_tail_features = cur_tail_feats
            all_outputs.append(cur_out)

        if self.out_cumulative:
            all_outputs_cum = [all_outputs[0]]
            for i in range(1, len(ms_features)):
                cur_out = all_outputs[i]
                cur_out_cum = cur_out + F.interpolate(all_outputs_cum[-1], size=cur_out.shape[2:],
                                                      mode='bilinear', align_corners=False)
                all_outputs_cum.append(cur_out_cum)
            all_outputs = all_outputs_cum

        if self.return_only_hr:
            return all_outputs[-1]
        else:
            return all_outputs[::-1]


class MultiscaleDiscriminatorSimple(nn.Module):
    def __init__(self, ms_impl):
        super().__init__()
        self.ms_impl = nn.ModuleList(ms_impl)

    @property
    def num_scales(self):
        return len(self.ms_impl)

    def forward(self, ms_inputs: List[torch.Tensor], smallest_scales_num: Optional[int] = None) \
            -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        :param ms_inputs: List of inputs of different resolutions from HR to LR
        :param smallest_scales_num: int or None, number of smallest scales to take at input
        :return: List of pairs (prediction, features) for different resolutions from HR to LR
        """
        if smallest_scales_num is None:
            assert len(self.ms_impl) == len(ms_inputs), (len(self.ms_impl), len(ms_inputs), smallest_scales_num)
            smallest_scales_num = len(self.heads)
        else:
            assert smallest_scales_num == len(ms_inputs) <= len(self.ms_impl), \
                (len(self.ms_impl), len(ms_inputs), smallest_scales_num)

        return [cur_discr(cur_input) for cur_discr, cur_input in zip(self.ms_impl[-smallest_scales_num:], ms_inputs)]


class SingleToMultiScaleInputMixin:
    def forward(self, x: torch.Tensor) -> List:
        orig_height, orig_width = x.shape[2:]
        factors = [2 ** i for i in range(self.num_scales)]
        ms_inputs = [F.interpolate(x, size=(orig_height // f, orig_width // f), mode='bilinear', align_corners=False)
                     for f in factors]
        return super().forward(ms_inputs)


class GeneratorMultiToSingleOutputMixin:
    def forward(self, x):
        return super().forward(x)[0]


class DiscriminatorMultiToSingleOutputMixin:
    def forward(self, x):
        out_feat_tuples = super().forward(x)
        return out_feat_tuples[0][0], [f for _, flist in out_feat_tuples for f in flist]


class DiscriminatorMultiToSingleOutputStackedMixin:
    def __init__(self, *args, return_feats_only_levels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_feats_only_levels = return_feats_only_levels

    def forward(self, x):
        out_feat_tuples = super().forward(x)
        outs = [out for out, _ in out_feat_tuples]
        scaled_outs = [outs[0]] + [F.interpolate(cur_out, size=outs[0].shape[-2:],
                                                 mode='bilinear', align_corners=False)
                                   for cur_out in outs[1:]]
        out = torch.cat(scaled_outs, dim=1)
        if self.return_feats_only_levels is not None:
            feat_lists = [out_feat_tuples[i][1] for i in self.return_feats_only_levels]
        else:
            feat_lists = [flist for _, flist in out_feat_tuples]
        feats = [f for flist in feat_lists for f in flist]
        return out, feats


class MultiscaleDiscrSingleInput(SingleToMultiScaleInputMixin, DiscriminatorMultiToSingleOutputStackedMixin, MultiscaleDiscriminatorSimple):
    pass


class MultiscaleResNetSingle(GeneratorMultiToSingleOutputMixin, SingleToMultiScaleInputMixin, MultiscaleResNet):
    pass
