#!/usr/bin/env python3
import numpy as np
from typing import List

from annotator.oneformer.detectron2.structures import Instances


def create_prediction_pairs(
    instances: Instances,
    prev_instances: Instances,
    iou_all: np.ndarray,
    threshold: float = 0.5,
) -> List:
    """
    Args:
        instances: predictions from current frame
        prev_instances: predictions from previous frame
        iou_all: 2D numpy array containing iou for each bbox pair
        threshold: below the threshold, doesn't consider the pair of bbox is valid
    Return:
        List of bbox pairs
    """
    bbox_pairs = []
    for i in range(len(instances)):
        for j in range(len(prev_instances)):
            if iou_all[i, j] < threshold:
                continue
            bbox_pairs.append(
                {
                    "idx": i,
                    "prev_idx": j,
                    "prev_id": prev_instances.ID[j],
                    "IoU": iou_all[i, j],
                    "prev_period": prev_instances.ID_period[j],
                }
            )
    return bbox_pairs


LARGE_COST_VALUE = 100000
