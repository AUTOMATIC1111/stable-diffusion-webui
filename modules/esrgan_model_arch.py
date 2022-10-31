# this file is adapted from https://github.com/victorca25/iNNfer

import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


####################
# RRDBNet Generator
####################

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nr=3, gc=32, upscale=4, norm_type=None,
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D',
            finalact=None, gaussian_noise=False, plus=False):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.resrgan_scale = 0
        if in_nc % 16 == 0:
            self.resrgan_scale = 1
        elif in_nc != 4 and in_nc % 4 == 0:
            self.resrgan_scale = 2

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)
        rb_blocks = [RRDB(nf, nr, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=norm_type, act_type=act_type, mode='CNA', convtype=convtype,
            gaussian_noise=gaussian_noise, plus=plus) for _ in range(nb)]
        LR_conv = conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode, convtype=convtype)

        if upsample_mode == 'upconv':
            upsample_block = upconv_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, convtype=convtype)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, convtype=convtype) for _ in range(n_upscale)]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type, convtype=convtype)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, convtype=convtype)

        outact = act(finalact) if finalact else None

        self.model = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler, HR_conv0, HR_conv1, outact)

    def forward(self, x, outm=None):
        if self.resrgan_scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        elif self.resrgan_scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        else:
            feat = x

        return self.model(feat)


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(self, nf, nr=3, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D',
            spectral_norm=False, gaussian_noise=False, plus=False):
        super(RRDB, self).__init__()
        # This is for backwards compatibility with existing models
        if nr == 3:
            self.RDB1 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB2 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
            self.RDB3 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                    norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise, plus=plus)
        else:
            RDB_list = [ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias, pad_type,
                                              norm_type, act_type, mode, convtype, spectral_norm=spectral_norm,
                                              gaussian_noise=gaussian_noise, plus=plus) for _ in range(nr)]
            self.RDBs = nn.Sequential(*RDB_list)

    def forward(self, x):
        if hasattr(self, 'RDB1'):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
        else:
            out = self.RDBs(x)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C. 
            {Rakotonirina} and A. {Rasoanaivo}
    """

    def __init__(self, nf=64, kernel_size=3, gc=32, stride=1, bias=1, pad_type='zero',
            norm_type=None, act_type='leakyrelu', mode='CNA', convtype='Conv2D',
            spectral_norm=False, gaussian_noise=False, plus=False):
        super(ResidualDenseBlock_5C, self).__init__()

        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(nf, gc) if plus else None

        self.conv1 = conv_block(nf, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        self.conv2 = conv_block(nf+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        self.conv3 = conv_block(nf+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        self.conv4 = conv_block(nf+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act_type, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nf+4*gc, nf, 3, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=last_act, mode=mode, convtype=convtype,
            spectral_norm=spectral_norm)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            return x5 * 0.2 + x


####################
# ESRGANplus
####################

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float)

    def forward(self, x):
        if self.training and self.sigma != 0:
            self.noise = self.noise.to(x.device)
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################
# SRVGGNetCompact
####################

class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    This class is copied from https://github.com/xinntao/Real-ESRGAN
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


####################
# Upsampler
####################

class Upsample(nn.Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                        pad_type='zero', norm_type=None, act_type='relu', convtype='Conv2D'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=None, act_type=None, convtype=convtype)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest', convtype='Conv2D'):
    """ Upconv layer """
    upscale_factor = (1, upscale_factor, upscale_factor) if convtype == 'Conv3D' else upscale_factor
    upsample = Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type, convtype=convtype)
    return sequential(upsample, conv)








####################
# Basic blocks
####################


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block. (block)
        num_basic_block (int): number of blocks. (n_layers)
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    """ activation helper """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type in ('leakyrelu', 'lrelu'):
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'tanh':  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == 'sigmoid':  # [0, 1] range output
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class Identity(nn.Module):
    def __init__(self, *kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *kwargs):
        return x


def norm(norm_type, nc):
    """ Return a normalization layer """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    """ padding layer helper """
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    """ Elementwise sum the output of a submodule to its input """
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        return 'Identity + \n|' + self.sub.__repr__().replace('\n', '\n|')


def sequential(*args):
    """ Flatten Sequential. It unwraps nn.Sequential. """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', convtype='Conv2D',
               spectral_norm=False):
    """ Conv layer with padding, normalization, activation """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    if convtype=='PartialConv2D':
        c = PartialConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
               dilation=dilation, bias=bias, groups=groups)
    elif convtype=='DeformConv2D':
        c = DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
               dilation=dilation, bias=bias, groups=groups)
    elif convtype=='Conv3D':
        c = nn.Conv3d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias, groups=groups)
    else:
        c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias, groups=groups)

    if spectral_norm:
        c = nn.utils.spectral_norm(c)

    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
