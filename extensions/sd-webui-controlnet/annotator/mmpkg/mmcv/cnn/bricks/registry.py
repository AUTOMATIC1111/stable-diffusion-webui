# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.utils import Registry

CONV_LAYERS = Registry('conv layer')
NORM_LAYERS = Registry('norm layer')
ACTIVATION_LAYERS = Registry('activation layer')
PADDING_LAYERS = Registry('padding layer')
UPSAMPLE_LAYERS = Registry('upsample layer')
PLUGIN_LAYERS = Registry('plugin layer')

DROPOUT_LAYERS = Registry('drop out layers')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')
