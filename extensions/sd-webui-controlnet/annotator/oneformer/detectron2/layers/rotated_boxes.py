# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import absolute_import, division, print_function, unicode_literals
import torch


def pairwise_iou_rotated(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in
    (x_center, y_center, width, height, angle) format.

    Arguments:
        boxes1 (Tensor[N, 5])
        boxes2 (Tensor[M, 5])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    return torch.ops.detectron2.box_iou_rotated(boxes1, boxes2)
