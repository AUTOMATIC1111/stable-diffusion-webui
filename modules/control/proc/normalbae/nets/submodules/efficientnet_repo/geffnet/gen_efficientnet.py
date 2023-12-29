""" Generic Efficient Networks

A generic MobileNet class with building blocks to support a variety of models:

* EfficientNet (B0-B8, L2 + Tensorflow pretrained AutoAug/RandAug/AdvProp/NoisyStudent ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665
  - Self-training with Noisy Student improves ImageNet classification - https://arxiv.org/abs/1911.04252

* EfficientNet-Lite

* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595

* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626

* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443

* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877

* And likely more...

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch.nn as nn
import torch.nn.functional as F

from .config import layer_config_kwargs, is_scriptable
from .conv2d_layers import select_conv2d
from .helpers import load_pretrained
from .efficientnet_builder import *

__all__ = ['GenEfficientNet', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_b1', 'mnasnet_140',
           'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'mnasnet_a1', 'semnasnet_140', 'mnasnet_small',
           'mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d',
           'fbnetc_100', 'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',  'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8',
           'efficientnet_l2', 'efficientnet_es', 'efficientnet_em', 'efficientnet_el',
           'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e',
           'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4',
           'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3',
           'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7', 'tf_efficientnet_b8',
           'tf_efficientnet_b0_ap', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b3_ap',
           'tf_efficientnet_b4_ap', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap',
           'tf_efficientnet_b8_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns',
           'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6_ns',
           'tf_efficientnet_b7_ns', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475',
           'tf_efficientnet_es', 'tf_efficientnet_em', 'tf_efficientnet_el',
           'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e',
           'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3',
           'tf_efficientnet_lite4',
           'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l']


model_urls = {
    'mnasnet_050': None,
    'mnasnet_075': None,
    'mnasnet_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth',
    'mnasnet_140': None,
    'mnasnet_small': None,

    'semnasnet_050': None,
    'semnasnet_075': None,
    'semnasnet_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth',
    'semnasnet_140': None,

    'mobilenetv2_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_100_ra-b33bc2c4.pth',
    'mobilenetv2_110d':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_110d_ra-77090ade.pth',
    'mobilenetv2_120d':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_120d_ra-5987e2ed.pth',
    'mobilenetv2_140':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv2_140_ra-21a4e913.pth',

    'fbnetc_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
    'spnasnet_100':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',

    'efficientnet_b0':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
    'efficientnet_b1':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
    'efficientnet_b2':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
    'efficientnet_b3':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
    'efficientnet_b4': None,
    'efficientnet_b5': None,
    'efficientnet_b6': None,
    'efficientnet_b7': None,
    'efficientnet_b8': None,
    'efficientnet_l2': None,

    'efficientnet_es':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_es_ra-f111e99c.pth',
    'efficientnet_em': None,
    'efficientnet_el': None,

    'efficientnet_cc_b0_4e': None,
    'efficientnet_cc_b0_8e': None,
    'efficientnet_cc_b1_8e': None,

    'efficientnet_lite0': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth',
    'efficientnet_lite1': None,
    'efficientnet_lite2': None,
    'efficientnet_lite3': None,
    'efficientnet_lite4': None,

    'tf_efficientnet_b0':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
    'tf_efficientnet_b1':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
    'tf_efficientnet_b2':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
    'tf_efficientnet_b3':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
    'tf_efficientnet_b4':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
    'tf_efficientnet_b5':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
    'tf_efficientnet_b6':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
    'tf_efficientnet_b7':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
    'tf_efficientnet_b8':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ra-572d5dd9.pth',

    'tf_efficientnet_b0_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
    'tf_efficientnet_b1_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
    'tf_efficientnet_b2_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
    'tf_efficientnet_b3_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
    'tf_efficientnet_b4_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
    'tf_efficientnet_b5_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
    'tf_efficientnet_b6_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
    'tf_efficientnet_b7_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
    'tf_efficientnet_b8_ap':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',

    'tf_efficientnet_b0_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ns-c0e6a31c.pth',
    'tf_efficientnet_b1_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ns-99dd0c41.pth',
    'tf_efficientnet_b2_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth',
    'tf_efficientnet_b3_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth',
    'tf_efficientnet_b4_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth',
    'tf_efficientnet_b5_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth',
    'tf_efficientnet_b6_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth',
    'tf_efficientnet_b7_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth',
    'tf_efficientnet_l2_ns_475':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns_475-bebbd00a.pth',
    'tf_efficientnet_l2_ns':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth',

    'tf_efficientnet_es':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
    'tf_efficientnet_em':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
    'tf_efficientnet_el':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',

    'tf_efficientnet_cc_b0_4e':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth',
    'tf_efficientnet_cc_b0_8e':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth',
    'tf_efficientnet_cc_b1_8e':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth',

    'tf_efficientnet_lite0':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth',
    'tf_efficientnet_lite1':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite1-bde8b488.pth',
    'tf_efficientnet_lite2':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite2-dcccb7df.pth',
    'tf_efficientnet_lite3':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite3-b733e338.pth',
    'tf_efficientnet_lite4':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite4-741542c3.pth',

    'mixnet_s': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth',
    'mixnet_m': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth',
    'mixnet_l': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth',
    'mixnet_xl': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl_ra-aac3c00c.pth',

    'tf_mixnet_s':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth',
    'tf_mixnet_m':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth',
    'tf_mixnet_l':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth',
}


class GenEfficientNet(nn.Module):
    """ Generic EfficientNets

    An implementation of mobile optimized networks that covers:
      * EfficientNet (B0-B8, L2, CondConv, EdgeTPU)
      * MixNet (Small, Medium, and Large, XL)
      * MNASNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, num_features=1280, stem_size=32, fix_stem=False,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        self.drop_rate = drop_rate

        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size

        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min,
            pad_type, act_layer, se_kwargs, norm_layer, norm_kwargs, drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs

        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for n, m in self.named_modules():
            if weight_init == 'goog':
                initialize_weight_goog(m, n)
            else:
                initialize_weight_default(m, n)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([
            self.conv_head, self.bn2, self.act2,
            self.global_pool, nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_model(model_kwargs, variant, pretrained=False):
    as_sequential = model_kwargs.pop('as_sequential', False)
    model = GenEfficientNet(**model_kwargs)
    if pretrained:
        load_pretrained(model, model_urls[variant])
    if as_sequential:
        model = model.as_sequential()
    return model


def _gen_mnasnet_a1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mnasnet_b1(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mnasnet_small(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        ['ds_r1_k3_s1_c8'],
        ['ir_r1_k3_s2_e3_c16'],
        ['ir_r2_k3_s2_e6_c16'],
        ['ir_r4_k5_s2_e6_c32_se0.25'],
        ['ir_r3_k3_s1_e6_c32_se0.25'],
        ['ir_r3_k5_s2_e6_c88_se0.25'],
        ['ir_r1_k3_s1_e6_c144']
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            stem_size=8,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
            num_features=1280 if fix_stem_head else round_channels(1280, channel_multiplier, 8, None),
            stem_size=32,
            fix_stem=fix_stem_head,
            channel_multiplier=channel_multiplier,
            norm_kwargs=resolve_bn_args(kwargs),
            act_layer=nn.ReLU6,
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_fbnetc(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            stem_size=16,
            num_features=1984,  # paper suggests this, but is not 100% clear
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_spnasnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier),
            num_features=round_channels(1280, channel_multiplier, 8, None),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'swish'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_efficientnet_edge(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier),
            num_features=round_channels(1280, channel_multiplier, 8, None),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_efficientnet_condconv(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=1, pretrained=False, **kwargs):
    """Creates an efficientnet-condconv model."""
    arch_def = [
      ['ds_r1_k3_s1_e1_c16_se0.25'],
      ['ir_r2_k3_s2_e6_c24_se0.25'],
      ['ir_r2_k5_s2_e6_c40_se0.25'],
      ['ir_r3_k3_s2_e6_c80_se0.25'],
      ['ir_r3_k5_s1_e6_c112_se0.25_cc4'],
      ['ir_r4_k5_s2_e6_c192_se0.25_cc4'],
      ['ir_r1_k3_s1_e6_c320_se0.25_cc4'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier, experts_multiplier=experts_multiplier),
            num_features=round_channels(1280, channel_multiplier, 8, None),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'swish'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_efficientnet_lite(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
            num_features=1280,
            stem_size=32,
            fix_stem=True,
            channel_multiplier=channel_multiplier,
            act_layer=nn.ReLU6,
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs,
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mixnet_s(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def),
            num_features=1536,
            stem_size=16,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def _gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
            num_features=1536,
            stem_size=24,
            channel_multiplier=channel_multiplier,
            act_layer=resolve_act_layer(kwargs, 'relu'),
            norm_kwargs=resolve_bn_args(kwargs),
            **kwargs
        )
        model = _create_model(model_kwargs, variant, pretrained)
    return model


def mnasnet_050(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.5. """
    model = _gen_mnasnet_b1('mnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


def mnasnet_075(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 0.75. """
    model = _gen_mnasnet_b1('mnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def mnasnet_100(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    model = _gen_mnasnet_b1('mnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mnasnet_b1(pretrained=False, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    return mnasnet_100(pretrained, **kwargs)


def mnasnet_140(pretrained=False, **kwargs):
    """ MNASNet B1,  depth multiplier of 1.4 """
    model = _gen_mnasnet_b1('mnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


def semnasnet_050(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 0.5 """
    model = _gen_mnasnet_a1('semnasnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


def semnasnet_075(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE),  depth multiplier of 0.75. """
    model = _gen_mnasnet_a1('semnasnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


def semnasnet_100(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    model = _gen_mnasnet_a1('semnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mnasnet_a1(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    return semnasnet_100(pretrained, **kwargs)


def semnasnet_140(pretrained=False, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.4. """
    model = _gen_mnasnet_a1('semnasnet_140', 1.4, pretrained=pretrained, **kwargs)
    return model


def mnasnet_small(pretrained=False, **kwargs):
    """ MNASNet Small,  depth multiplier of 1.0. """
    model = _gen_mnasnet_small('mnasnet_small', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv2_100(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.0 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def mobilenetv2_140(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.4 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_140', 1.4, pretrained=pretrained, **kwargs)
    return model


def mobilenetv2_110d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.1 channel, 1.2 depth multipliers"""
    model = _gen_mobilenet_v2(
        'mobilenetv2_110d', 1.1, depth_multiplier=1.2, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


def mobilenetv2_120d(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 1.2 channel, 1.4 depth multipliers """
    model = _gen_mobilenet_v2(
        'mobilenetv2_120d', 1.2, depth_multiplier=1.4, fix_stem_head=True, pretrained=pretrained, **kwargs)
    return model


def fbnetc_100(pretrained=False, **kwargs):
    """ FBNet-C """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_fbnetc('fbnetc_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def spnasnet_100(pretrained=False, **kwargs):
    """ Single-Path NAS Pixel1"""
    model = _gen_spnasnet('spnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train set drop_rate=0.2, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1 """
    # NOTE for train set drop_rate=0.2, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2 """
    # NOTE for train set drop_rate=0.3, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3 """
    # NOTE for train set drop_rate=0.3, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4 """
    # NOTE for train set drop_rate=0.4, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5 """
    # NOTE for train set drop_rate=0.4, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6 """
    # NOTE for train set drop_rate=0.5, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7 """
    # NOTE for train set drop_rate=0.5, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


def efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8 """
    # NOTE for train set drop_rate=0.5, drop_connect_rate=0.2
    model = _gen_efficientnet(
        'efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


def efficientnet_l2(pretrained=False, **kwargs):
    """ EfficientNet-L2. """
    # NOTE for train, drop_rate should be 0.5
    model = _gen_efficientnet(
        'efficientnet_l2', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


def efficientnet_es(pretrained=False, **kwargs):
    """ EfficientNet-Edge Small. """
    model = _gen_efficientnet_edge(
        'efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def efficientnet_em(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Medium. """
    model = _gen_efficientnet_edge(
        'efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def efficientnet_el(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Large. """
    model = _gen_efficientnet_edge(
        'efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def efficientnet_cc_b0_4e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train set drop_rate=0.25, drop_connect_rate=0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def efficientnet_cc_b0_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    # NOTE for train set drop_rate=0.25, drop_connect_rate=0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


def efficientnet_cc_b1_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B1 w/ 8 Experts """
    # NOTE for train set drop_rate=0.25, drop_connect_rate=0.2
    model = _gen_efficientnet_condconv(
        'efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


def efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    model = _gen_efficientnet_lite(
        'efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1 """
    model = _gen_efficientnet_lite(
        'efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2 """
    model = _gen_efficientnet_lite(
        'efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3 """
    model = _gen_efficientnet_lite(
        'efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4 """
    model = _gen_efficientnet_lite(
        'efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b0(pretrained=False, **kwargs):
    """ EfficientNet-B0 AutoAug. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b1(pretrained=False, **kwargs):
    """ EfficientNet-B1 AutoAug. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b2(pretrained=False, **kwargs):
    """ EfficientNet-B2 AutoAug. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b3(pretrained=False, **kwargs):
    """ EfficientNet-B3 AutoAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b4(pretrained=False, **kwargs):
    """ EfficientNet-B4 AutoAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b5(pretrained=False, **kwargs):
    """ EfficientNet-B5 RandAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b6(pretrained=False, **kwargs):
    """ EfficientNet-B6 AutoAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b7(pretrained=False, **kwargs):
    """ EfficientNet-B7 RandAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b8(pretrained=False, **kwargs):
    """ EfficientNet-B8 RandAug. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b0_ap(pretrained=False, **kwargs):
    """ EfficientNet-B0 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0_ap', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b1_ap(pretrained=False, **kwargs):
    """ EfficientNet-B1 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1_ap', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b2_ap(pretrained=False, **kwargs):
    """ EfficientNet-B2 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2_ap', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b3_ap(pretrained=False, **kwargs):
    """ EfficientNet-B3 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3_ap', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b4_ap(pretrained=False, **kwargs):
    """ EfficientNet-B4 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4_ap', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b5_ap(pretrained=False, **kwargs):
    """ EfficientNet-B5 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5_ap', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b6_ap(pretrained=False, **kwargs):
    """ EfficientNet-B6 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6_ap', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b7_ap(pretrained=False, **kwargs):
    """ EfficientNet-B7 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ap', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b8_ap(pretrained=False, **kwargs):
    """ EfficientNet-B8 AdvProp. Tensorflow compatible variant
    Paper: Adversarial Examples Improve Image Recognition (https://arxiv.org/abs/1911.09665)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b8_ap', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b0_ns(pretrained=False, **kwargs):
    """ EfficientNet-B0 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b0_ns', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b1_ns(pretrained=False, **kwargs):
    """ EfficientNet-B1 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b1_ns', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b2_ns(pretrained=False, **kwargs):
    """ EfficientNet-B2 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b2_ns', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b3_ns(pretrained=False, **kwargs):
    """ EfficientNet-B3 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b3_ns', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b4_ns(pretrained=False, **kwargs):
    """ EfficientNet-B4 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b4_ns', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b5_ns(pretrained=False, **kwargs):
    """ EfficientNet-B5 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b5_ns', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b6_ns(pretrained=False, **kwargs):
    """ EfficientNet-B6 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b6_ns', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_b7_ns(pretrained=False, **kwargs):
    """ EfficientNet-B7 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ns', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_l2_ns_475(pretrained=False, **kwargs):
    """ EfficientNet-L2 NoisyStudent @ 475x475. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_l2_ns_475', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_l2_ns(pretrained=False, **kwargs):
    """ EfficientNet-L2 NoisyStudent. Tensorflow compatible variant
    Paper: Self-training with Noisy Student improves ImageNet classification (https://arxiv.org/abs/1911.04252)
    """
    # NOTE for train, drop_rate should be 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet(
        'tf_efficientnet_l2_ns', channel_multiplier=4.3, depth_multiplier=5.3, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_es(pretrained=False, **kwargs):
    """ EfficientNet-Edge Small. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_es', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_em(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_el(pretrained=False, **kwargs):
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_edge(
        'tf_efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_cc_b0_4e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 4 Experts """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_cc_b0_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_cc_b1_8e(pretrained=False, **kwargs):
    """ EfficientNet-CondConv-B1 w/ 8 Experts """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_lite1(pretrained=False, **kwargs):
    """ EfficientNet-Lite1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_lite2(pretrained=False, **kwargs):
    """ EfficientNet-Lite2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_lite3(pretrained=False, **kwargs):
    """ EfficientNet-Lite3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return model


def tf_efficientnet_lite4(pretrained=False, **kwargs):
    """ EfficientNet-Lite4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return model


def mixnet_s(pretrained=False, **kwargs):
    """Creates a MixNet Small model.
    """
    # NOTE for train set drop_rate=0.2
    model = _gen_mixnet_s(
        'mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def mixnet_m(pretrained=False, **kwargs):
    """Creates a MixNet Medium model.
    """
    # NOTE for train set drop_rate=0.25
    model = _gen_mixnet_m(
        'mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def mixnet_l(pretrained=False, **kwargs):
    """Creates a MixNet Large model.
    """
    # NOTE for train set drop_rate=0.25
    model = _gen_mixnet_m(
        'mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


def mixnet_xl(pretrained=False, **kwargs):
    """Creates a MixNet Extra-Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    # NOTE for train set drop_rate=0.25, drop_connect_rate=0.2
    model = _gen_mixnet_m(
        'mixnet_xl', channel_multiplier=1.6, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return model


def mixnet_xxl(pretrained=False, **kwargs):
    """Creates a MixNet Double Extra Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    # NOTE for train set drop_rate=0.3, drop_connect_rate=0.2
    model = _gen_mixnet_m(
        'mixnet_xxl', channel_multiplier=2.4, depth_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model


def tf_mixnet_s(pretrained=False, **kwargs):
    """Creates a MixNet Small model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_s(
        'tf_mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mixnet_m(pretrained=False, **kwargs):
    """Creates a MixNet Medium model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


def tf_mixnet_l(pretrained=False, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mixnet_m(
        'tf_mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return model
