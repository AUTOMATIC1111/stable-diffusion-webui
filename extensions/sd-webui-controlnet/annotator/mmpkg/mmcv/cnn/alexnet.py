# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet backbone.

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes=-1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            from ..runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)

        return x
