# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn


class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()
