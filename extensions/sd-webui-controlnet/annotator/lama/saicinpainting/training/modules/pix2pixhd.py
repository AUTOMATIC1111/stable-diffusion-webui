# original: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
import collections
from functools import partial
import functools
import logging
from collections import defaultdict

import numpy as np
import torch.nn as nn

from annotator.lama.saicinpainting.training.modules.base import BaseDiscriminator, deconv_factory, get_conv_block_ctor, get_norm_layer, get_activation
from annotator.lama.saicinpainting.training.modules.ffc import FFCResnetBlock
from annotator.lama.saicinpainting.training.modules.multidilated_conv import MultidilatedConv

class DotDict(defaultdict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """dot.notation access to dictionary attributes"""
    __getattr__ = defaultdict.get
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, conv_kind='default',
                 dilation=1, in_dim=None, groups=1, second_dilation=None):
        super(ResnetBlock, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout,
                                                conv_kind=conv_kind, dilation=dilation, in_dim=in_dim, groups=groups,
                                                second_dilation=second_dilation)

        if self.in_dim is not None:
            self.input_conv = nn.Conv2d(in_dim, dim, 1)

        self.out_channnels = dim

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default',
                         dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = get_conv_block_ctor(conv_kind)

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == 'zero':
            p = dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if in_dim is None:
            in_dim = dim

        conv_block += [conv_layer(in_dim, dim, kernel_size=3, padding=p, dilation=dilation),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation)]
        elif padding_type == 'zero':
            p = second_dilation
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding=p, dilation=second_dilation, groups=groups),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        x_before = x
        if self.in_dim is not None:
            x = self.input_conv(x)
        out = x + self.conv_block(x_before)
        return out

class ResnetBlock5x5(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, conv_kind='default',
                 dilation=1, in_dim=None, groups=1, second_dilation=None):
        super(ResnetBlock5x5, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        if second_dilation is None:
            second_dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout,
                                                conv_kind=conv_kind, dilation=dilation, in_dim=in_dim, groups=groups,
                                                second_dilation=second_dilation)

        if self.in_dim is not None:
            self.input_conv = nn.Conv2d(in_dim, dim, 1)

        self.out_channnels = dim

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, conv_kind='default',
                         dilation=1, in_dim=None, groups=1, second_dilation=1):
        conv_layer = get_conv_block_ctor(conv_kind)

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(dilation * 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(dilation * 2)]
        elif padding_type == 'zero':
            p = dilation * 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if in_dim is None:
            in_dim = dim

        conv_block += [conv_layer(in_dim, dim, kernel_size=5, padding=p, dilation=dilation),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(second_dilation * 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(second_dilation * 2)]
        elif padding_type == 'zero':
            p = second_dilation * 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [conv_layer(dim, dim, kernel_size=5, padding=p, dilation=second_dilation, groups=groups),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        x_before = x
        if self.in_dim is not None:
            x = self.input_conv(x)
        out = x + self.conv_block(x_before)
        return out


class MultidilatedResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, conv_layer, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, conv_layer, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, conv_layer, norm_layer, activation, use_dropout, dilation=1):
        conv_block = []
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MultiDilatedGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 n_blocks=3, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default',
                 deconv_kind='convtranspose', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, affine=None, up_activation=nn.ReLU(True),
                 add_out_act=True, max_features=1024, multidilation_kwargs={},
                 ffc_positions=None, ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        conv_layer = get_conv_block_ctor(conv_kind)
        resnet_conv_layer = functools.partial(get_conv_block_ctor('multidilated'), **multidilation_kwargs)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)

        model = [nn.ReflectionPad2d(3),
                 conv_layer(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]

        identity = Identity()
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i

            model += [conv_layer(min(max_features, ngf * mult),
                                    min(max_features, ngf * mult * 2),
                                    kernel_size=3, stride=2, padding=1),
                        norm_layer(min(max_features, ngf * mult * 2)),
                        activation]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            if ffc_positions is not None and i in ffc_positions:
                model += [FFCResnetBlock(feats_num_bottleneck, padding_type, norm_layer, activation_layer=nn.ReLU,
                                         inline=True, **ffc_kwargs)]
            model += [MultidilatedResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                              conv_layer=resnet_conv_layer, activation=activation,
                                              norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += deconv_factory(deconv_kind, ngf, mult, up_norm_layer, up_activation, max_features)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ConfigGlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 n_blocks=3, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default',
                 deconv_kind='convtranspose', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, affine=None, up_activation=nn.ReLU(True),
                 add_out_act=True, max_features=1024,
                 manual_block_spec=[],
                 resnet_block_kind='multidilatedresnetblock',
                 resnet_conv_kind='multidilated',
                 resnet_dilation=1,
                 multidilation_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        conv_layer = get_conv_block_ctor(conv_kind)
        resnet_conv_layer = functools.partial(get_conv_block_ctor(resnet_conv_kind), **multidilation_kwargs)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)

        model = [nn.ReflectionPad2d(3),
                 conv_layer(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]

        identity = Identity()

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [conv_layer(min(max_features, ngf * mult),
                                    min(max_features, ngf * mult * 2),
                                    kernel_size=3, stride=2, padding=1),
                        norm_layer(min(max_features, ngf * mult * 2)),
                        activation]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        if len(manual_block_spec) == 0:
            manual_block_spec = [
                DotDict(lambda : None, {
                    'n_blocks': n_blocks,
                    'use_default': True})
            ]

        ### resnet blocks
        for block_spec in manual_block_spec:
            def make_and_add_blocks(model, block_spec):
                block_spec = DotDict(lambda : None, block_spec)
                if not block_spec.use_default:
                    resnet_conv_layer = functools.partial(get_conv_block_ctor(block_spec.resnet_conv_kind), **block_spec.multidilation_kwargs)
                    resnet_conv_kind = block_spec.resnet_conv_kind
                    resnet_block_kind = block_spec.resnet_block_kind
                    if block_spec.resnet_dilation is not None:
                        resnet_dilation = block_spec.resnet_dilation
                for i in range(block_spec.n_blocks):
                    if resnet_block_kind == "multidilatedresnetblock":
                        model += [MultidilatedResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                                        conv_layer=resnet_conv_layer, activation=activation,
                                                        norm_layer=norm_layer)]
                    if resnet_block_kind == "resnetblock":                                            
                        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                            conv_kind=resnet_conv_kind)]
                    if resnet_block_kind == "resnetblock5x5":                                            
                        model += [ResnetBlock5x5(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                            conv_kind=resnet_conv_kind)]
                    if resnet_block_kind == "resnetblockdwdil":
                        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                            conv_kind=resnet_conv_kind, dilation=resnet_dilation, second_dilation=resnet_dilation)]
            make_and_add_blocks(model, block_spec)
        
        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += deconv_factory(deconv_kind, ngf, mult, up_norm_layer, up_activation, max_features)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def make_dil_blocks(dilated_blocks_n, dilation_block_kind, dilated_block_kwargs):
    blocks = []
    for i in range(dilated_blocks_n):
        if dilation_block_kind == 'simple':
            blocks.append(ResnetBlock(**dilated_block_kwargs, dilation=2 ** (i + 1)))
        elif dilation_block_kind == 'multi':
            blocks.append(MultidilatedResnetBlock(**dilated_block_kwargs))
        else:
            raise ValueError(f'dilation_block_kind could not be "{dilation_block_kind}"')
    return blocks


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', conv_kind='default', activation=nn.ReLU(True),
                 up_norm_layer=nn.BatchNorm2d, affine=None,
                 up_activation=nn.ReLU(True), dilated_blocks_n=0, dilated_blocks_n_start=0,
                 dilated_blocks_n_middle=0,
                 add_out_act=True,
                 max_features=1024, is_resblock_depthwise=False,
                 ffc_positions=None, ffc_kwargs={}, dilation=1, second_dilation=None,
                 dilation_block_kind='simple', multidilation_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        conv_layer = get_conv_block_ctor(conv_kind)
        norm_layer = get_norm_layer(norm_layer)
        if affine is not None:
            norm_layer = partial(norm_layer, affine=affine)
        up_norm_layer = get_norm_layer(up_norm_layer)
        if affine is not None:
            up_norm_layer = partial(up_norm_layer, affine=affine)

        if ffc_positions is not None:
            ffc_positions = collections.Counter(ffc_positions)

        model = [nn.ReflectionPad2d(3),
                 conv_layer(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 activation]

        identity = Identity()
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i

            model += [conv_layer(min(max_features, ngf * mult),
                                min(max_features, ngf * mult * 2),
                                kernel_size=3, stride=2, padding=1),
                        norm_layer(min(max_features, ngf * mult * 2)),
                        activation]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        dilated_block_kwargs = dict(dim=feats_num_bottleneck, padding_type=padding_type,
                                    activation=activation, norm_layer=norm_layer)
        if dilation_block_kind == 'simple':
            dilated_block_kwargs['conv_kind'] = conv_kind
        elif dilation_block_kind == 'multi':
            dilated_block_kwargs['conv_layer'] = functools.partial(
                get_conv_block_ctor('multidilated'), **multidilation_kwargs)

        # dilated blocks at the start of the bottleneck sausage
        if dilated_blocks_n_start is not None and dilated_blocks_n_start > 0:
            model += make_dil_blocks(dilated_blocks_n_start, dilation_block_kind, dilated_block_kwargs)

        # resnet blocks
        for i in range(n_blocks):
            # dilated blocks at the middle of the bottleneck sausage
            if i == n_blocks // 2 and dilated_blocks_n_middle is not None and dilated_blocks_n_middle > 0:
                model += make_dil_blocks(dilated_blocks_n_middle, dilation_block_kind, dilated_block_kwargs)
            
            if ffc_positions is not None and i in ffc_positions:
                for _ in range(ffc_positions[i]):  # same position can occur more than once
                    model += [FFCResnetBlock(feats_num_bottleneck, padding_type, norm_layer, activation_layer=nn.ReLU,
                                             inline=True, **ffc_kwargs)]

            if is_resblock_depthwise:
                resblock_groups = feats_num_bottleneck
            else:
                resblock_groups = 1

            model += [ResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation=activation,
                                    norm_layer=norm_layer, conv_kind=conv_kind, groups=resblock_groups,
                                    dilation=dilation, second_dilation=second_dilation)]
            

        # dilated blocks at the end of the bottleneck sausage
        if dilated_blocks_n is not None and dilated_blocks_n > 0:
            model += make_dil_blocks(dilated_blocks_n, dilation_block_kind, dilated_block_kwargs)

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class GlobalGeneratorGated(GlobalGenerator):
    def __init__(self, *args, **kwargs):
        real_kwargs=dict(
            conv_kind='gated_bn_relu',
            activation=nn.Identity(),
            norm_layer=nn.Identity
        )
        real_kwargs.update(kwargs)
        super().__init__(*args, **real_kwargs)


class GlobalGeneratorFromSuperChannels(nn.Module):
    def __init__(self, input_nc, output_nc, n_downsampling, n_blocks, super_channels, norm_layer="bn", padding_type='reflect', add_out_act=True):
        super().__init__()
        self.n_downsampling = n_downsampling
        norm_layer = get_norm_layer(norm_layer)
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        channels = self.convert_super_channels(super_channels)
        self.channels = channels

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, channels[0], kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(channels[0]),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(channels[0+i], channels[1+i], kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(channels[1+i]),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            c = n_downsampling
            dim = channels[c]
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer)]

        for i in range(n_blocks2):
            c = n_downsampling+1
            dim = channels[c]
            kwargs = {}
            if i == 0:
                kwargs = {"in_dim": channels[c-1]}
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer, **kwargs)]

        for i in range(n_blocks3):
            c = n_downsampling+2
            dim = channels[c]
            kwargs = {}
            if i == 0:
                kwargs = {"in_dim": channels[c-1]}
            model += [ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer, **kwargs)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(channels[n_downsampling+3+i],
                                           channels[n_downsampling+3+i+1],
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                      norm_layer(channels[n_downsampling+3+i+1]),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(channels[2*n_downsampling+3], output_nc, kernel_size=7, padding=0)]

        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def convert_super_channels(self, super_channels):
        n_downsampling = self.n_downsampling
        result = []
        cnt = 0

        if n_downsampling == 2:
            N1 = 10
        elif n_downsampling == 3:
            N1 = 13
        else:
            raise NotImplementedError

        for i in range(0, N1):
            if i in [1,4,7,10]:
                channel = super_channels[cnt] * (2 ** cnt)
                config = {'channel': channel}
                result.append(channel)
                logging.info(f"Downsample channels {result[-1]}")
                cnt += 1

        for i in range(3):
            for counter, j in enumerate(range(N1 + i * 3, N1 + 3 + i * 3)):
                if len(super_channels) == 6:
                    channel = super_channels[3] * 4
                else:
                    channel = super_channels[i + 3] * 4
                config = {'channel': channel}
                if counter == 0:
                    result.append(channel)
                    logging.info(f"Bottleneck channels {result[-1]}")
        cnt = 2

        for i in range(N1+9, N1+21):
            if i in [22, 25,28]:
                cnt -= 1
                if len(super_channels) == 6:
                    channel = super_channels[5 - cnt] * (2 ** cnt)
                else:
                    channel = super_channels[7 - cnt] * (2 ** cnt)
                result.append(int(channel))
                logging.info(f"Upsample channels {result[-1]}")
        return result

    def forward(self, input):
        return self.model(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class MultidilatedNLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, multidilation_kwargs={}):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, padding=[2, 3], **multidilation_kwargs),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class NLayerDiscriminatorAsGen(NLayerDiscriminator):
    def forward(self, x):
        return super().forward(x)[0]
