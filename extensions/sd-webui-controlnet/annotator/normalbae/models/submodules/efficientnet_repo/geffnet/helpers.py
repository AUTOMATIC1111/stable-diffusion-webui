""" Checkpoint loading / state_dict helpers
Copyright 2020 Ross Wightman
"""
import torch
import os
from collections import OrderedDict
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("=> Loaded checkpoint '{}'".format(checkpoint_path))
    else:
        print("=> Error: No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(model, url, filter_fn=None, strict=True):
    if not url:
        print("=> Warning: Pretrained model URL is empty, using random initialization.")
        return

    state_dict = load_state_dict_from_url(url, progress=False, map_location='cpu')

    input_conv = 'conv_stem'
    classifier = 'classifier'
    in_chans = getattr(model, input_conv).weight.shape[1]
    num_classes = getattr(model, classifier).weight.shape[0]

    input_conv_weight = input_conv + '.weight'
    pretrained_in_chans = state_dict[input_conv_weight].shape[1]
    if in_chans != pretrained_in_chans:
        if in_chans == 1:
            print('=> Converting pretrained input conv {} from {} to 1 channel'.format(
                input_conv_weight, pretrained_in_chans))
            conv1_weight = state_dict[input_conv_weight]
            state_dict[input_conv_weight] = conv1_weight.sum(dim=1, keepdim=True)
        else:
            print('=> Discarding pretrained input conv {} since input channel count != {}'.format(
                input_conv_weight, pretrained_in_chans))
            del state_dict[input_conv_weight]
            strict = False

    classifier_weight = classifier + '.weight'
    pretrained_num_classes = state_dict[classifier_weight].shape[0]
    if num_classes != pretrained_num_classes:
        print('=> Discarding pretrained classifier since num_classes != {}'.format(pretrained_num_classes))
        del state_dict[classifier_weight]
        del state_dict[classifier + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)
