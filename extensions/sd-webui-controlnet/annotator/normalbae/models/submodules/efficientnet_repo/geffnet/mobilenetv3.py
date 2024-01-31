""" MobileNet-V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_act_fn, get_act_layer, HardSwish
from .config import layer_config_kwargs
from .conv2d_layers import select_conv2d
from .helpers import load_pretrained
from .efficientnet_builder import *

__all__ = ['mobilenetv3_rw', 'mobilenetv3_large_075', 'mobilenetv3_large_100', 'mobilenetv3_large_minimal_100',
           'mobilenetv3_small_075', 'mobilenetv3_small_100', 'mobilenetv3_small_minimal_100',
           'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100',
           'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100']

model_urls = {
    'mobilenetv3_rw':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
    'mobilenetv3_large_075': None,
    'mobilenetv3_large_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth',
    'mobilenetv3_large_minimal_100': None,
    'mobilenetv3_small_075': None,
    'mobilenetv3_small_100': None,
    'mobilenetv3_small_minimal_100': None,
    'tf_mobilenetv3_large_075':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
    'tf_mobilenetv3_large_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
    'tf_mobilenetv3_large_minimal_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
    'tf_mobilenetv3_small_075':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
    'tf_mobilenetv3_small_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
    'tf_mobilenetv3_small_minimal_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
}


class MobileNetV3(nn.Module):
    """ MobileNet-V3

    A this model utilizes the MobileNet-v3 specific 'efficient head', where global pooling is done before the
    head convolution without a final batch-norm layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16, num_features=1280, head_bias=True,
                 channel_multiplier=1.0, pad_type='', act_layer=HardSwish, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog'):
        super(MobileNetV3, self).__init__()
        self.drop_rate = drop_rate

        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, pad_type=pad_type, act_layer=act_layer, se_kwargs=se_kwargs,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, drop_connect_rate=drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if weight_init == 'goog':
                initialize_weight_goog(m)
            else:
                initialize_weight_default(m)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([
            self.global_pool, self.conv_head, self.act2,
            nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_model(model_kwargs, variant, pretrained=False):
    as_sequential = model_kwargs.pop('as_sequential', False)
    model = MobileNetV3(**model_kwargs)
    if pretrained and model_urls[variant]:
        load_pretrained(model, model_urls[variant])
    if as_sequential:
        model = model.as_sequential()
    return model


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model (RW variant).

    Paper: https://arxiv.org/abs/1905.02244

    This was my first attempt at reproducing the MobileNet-V3 from paper alone. It came close to the
    eventual Tensorflow reference impl but has a few differences:
    1. This model has no bias on the head convolution
    2. This model forces no residual (noskip) on the first DWS block, this is different than MnasNet
    3. This model always uses ReLU for the SE activation layer, other models in the family inherit their act layer
       from their parent block
    4. This model does not enforce divisible by 8 limitation on the SE reduction channel count

    Overall the changes are fairly minor and result in a very small parameter count difference and no
    top-1/5

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            head_bias=False,  # one of my mistakes
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'hard_swish'),
            se_kwargs=dict(gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 large/small/minimal models.

    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = 'hard_swish'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = 'hard_swish'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            num_features=num_features,
            stem_size=16,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, act_layer),
            se_kwargs=dict(
                act_layer=get_act_layer('relu'), gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True, divisor=8),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet-V3 RW
    Attn: See note in gen function for this variant.
    """
    # NOTE for train set drop_rate=0.2
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 Large 0.75"""
    # NOTE for train set drop_rate=0.2
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large 1.0 """
    # NOTE for train set drop_rate=0.2
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large (Minimalistic) 1.0 """
    # NOTE for train set drop_rate=0.2
    model = _gen_mobilenet_v3('mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 Small 0.75 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 Small 1.0 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 Small (Minimalistic) 1.0 """
    model = _gen_mobilenet_v3('mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 Large 0.75. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large 1.0. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large Minimalistic 1.0. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 Small 0.75. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 Small 1.0. Tensorflow compat variant."""
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 Small Minimalistic 1.0. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model
