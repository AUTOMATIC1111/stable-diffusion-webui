from argparse import Namespace

import os.path
import numpy as np
import cv2
from torchvision import transforms, models
import torch
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
import argparse

from torch.nn import functional as F

import models_cnx.convnext
from models_cnx.convnext import convnext_tiny as convnext_tiny_raw

@register_model
def convnext_tiny(pretrained=False,in_22k=False, pretrained_cfg=None, **kwargs):
    return convnext_tiny_raw(pretrained, in_22k, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ConvNextCfg={
    # Model parameters
    'model': 'convnext_tiny',
    'drop_path': 0.0,
    'input_size': 384,
    # Evaluation parameters
    'crop_pct': None,
    # Dataset parameters
    'nb_classes': 3,
    'imagenet_default_mean_and_std': True,
}

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

class XPDiscriminator():
    def __init__(self, ckpt) -> None:
        args = Namespace()
        for k,v in ConvNextCfg.items():
            setattr(args, k, v)
        self.args=args

        self.net = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
        # net.load_state_dict(torch.load(args.ckpt)['model'])
        load_state_dict(self.net, torch.load(ckpt, map_location='cpu'), prefix='')
        self.net = self.net.to(device)
        self.net.eval()

        self.img_mean=torch.tensor(IMAGENET_DEFAULT_MEAN).view(1,-1,1,1).to(device)
        self.img_std=torch.tensor(IMAGENET_DEFAULT_STD).view(1,-1,1,1).to(device)

    def get_score(self, img):
        img = ((img+1.)/2.).sub(self.img_mean).div(self.img_std)

        img=F.interpolate(img, size=(self.args.input_size, self.args.input_size), mode='bicubic', align_corners=True)
        pred = self.net(img)
        return torch.softmax(pred, dim=-1)[:,1]

    def get_all(self, img):
        img = ((img + 1.) / 2.).sub(self.img_mean).div(self.img_std)

        img = F.interpolate(img, size=(self.args.input_size, self.args.input_size), mode='bicubic', align_corners=True)
        pred = self.net(img)
        return pred