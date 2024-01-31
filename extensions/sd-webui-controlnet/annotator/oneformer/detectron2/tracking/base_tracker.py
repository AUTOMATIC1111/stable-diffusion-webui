#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.utils.registry import Registry

from ..config.config import CfgNode as CfgNode_
from ..structures import Instances

TRACKER_HEADS_REGISTRY = Registry("TRACKER_HEADS")
TRACKER_HEADS_REGISTRY.__doc__ = """
Registry for tracking classes.
"""


class BaseTracker(object):
    """
    A parent class for all trackers
    """

    @configurable
    def __init__(self, **kwargs):
        self._prev_instances = None  # (D2)instances for previous frame
        self._matched_idx = set()  # indices in prev_instances found matching
        self._matched_ID = set()  # idendities in prev_instances found matching
        self._untracked_prev_idx = set()  # indices in prev_instances not found matching
        self._id_count = 0  # used to assign new id

    @classmethod
    def from_config(cls, cfg: CfgNode_):
        raise NotImplementedError("Calling BaseTracker::from_config")

    def update(self, predictions: Instances) -> Instances:
        """
        Args:
            predictions: D2 Instances for predictions of the current frame
        Return:
            D2 Instances for predictions of the current frame with ID assigned

        _prev_instances and instances will have the following fields:
          .pred_boxes               (shape=[N, 4])
          .scores                   (shape=[N,])
          .pred_classes             (shape=[N,])
          .pred_keypoints           (shape=[N, M, 3], Optional)
          .pred_masks               (shape=List[2D_MASK], Optional)   2D_MASK: shape=[H, W]
          .ID                       (shape=[N,])

        N: # of detected bboxes
        H and W: height and width of 2D mask
        """
        raise NotImplementedError("Calling BaseTracker::update")


def build_tracker_head(cfg: CfgNode_) -> BaseTracker:
    """
    Build a tracker head from `cfg.TRACKER_HEADS.TRACKER_NAME`.

    Args:
        cfg: D2 CfgNode, config file with tracker information
    Return:
        tracker object
    """
    name = cfg.TRACKER_HEADS.TRACKER_NAME
    tracker_class = TRACKER_HEADS_REGISTRY.get(name)
    return tracker_class(cfg)
