from argparse import Namespace

import os.path
import numpy as np
import cv2
from torchvision import transforms, models
import torch
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import argparse
from . import utils
from torch.nn import functional as F

import modules.conv_next.models.convnext
import modules.conv_next.models.convnext_isotropic

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
    'ckpt': 'models/convnext/checkpoint-best_t5.pth',
}

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
        utils.load_state_dict(self.net, torch.load(ckpt, map_location='cpu'), prefix='')
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