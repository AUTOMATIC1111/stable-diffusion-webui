#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from typing import List

from annotator.oneformer.detectron2.config import CfgNode as CfgNode_
from annotator.oneformer.detectron2.config import configurable

from .base_tracker import TRACKER_HEADS_REGISTRY
from .vanilla_hungarian_bbox_iou_tracker import VanillaHungarianBBoxIOUTracker


@TRACKER_HEADS_REGISTRY.register()
class IOUWeightedHungarianBBoxIOUTracker(VanillaHungarianBBoxIOUTracker):
    """
    A tracker using IoU as weight in Hungarian algorithm, also known
    as Munkres or Kuhn-Munkres algorithm
    """

    @configurable
    def __init__(
        self,
        *,
        video_height: int,
        video_width: int,
        max_num_instances: int = 200,
        max_lost_frame_count: int = 0,
        min_box_rel_dim: float = 0.02,
        min_instance_period: int = 1,
        track_iou_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Args:
        video_height: height the video frame
        video_width: width of the video frame
        max_num_instances: maximum number of id allowed to be tracked
        max_lost_frame_count: maximum number of frame an id can lost tracking
                              exceed this number, an id is considered as lost
                              forever
        min_box_rel_dim: a percentage, smaller than this dimension, a bbox is
                         removed from tracking
        min_instance_period: an instance will be shown after this number of period
                             since its first showing up in the video
        track_iou_threshold: iou threshold, below this number a bbox pair is removed
                             from tracking
        """
        super().__init__(
            video_height=video_height,
            video_width=video_width,
            max_num_instances=max_num_instances,
            max_lost_frame_count=max_lost_frame_count,
            min_box_rel_dim=min_box_rel_dim,
            min_instance_period=min_instance_period,
            track_iou_threshold=track_iou_threshold,
        )

    @classmethod
    def from_config(cls, cfg: CfgNode_):
        """
        Old style initialization using CfgNode

        Args:
            cfg: D2 CfgNode, config file
        Return:
            dictionary storing arguments for __init__ method
        """
        assert "VIDEO_HEIGHT" in cfg.TRACKER_HEADS
        assert "VIDEO_WIDTH" in cfg.TRACKER_HEADS
        video_height = cfg.TRACKER_HEADS.get("VIDEO_HEIGHT")
        video_width = cfg.TRACKER_HEADS.get("VIDEO_WIDTH")
        max_num_instances = cfg.TRACKER_HEADS.get("MAX_NUM_INSTANCES", 200)
        max_lost_frame_count = cfg.TRACKER_HEADS.get("MAX_LOST_FRAME_COUNT", 0)
        min_box_rel_dim = cfg.TRACKER_HEADS.get("MIN_BOX_REL_DIM", 0.02)
        min_instance_period = cfg.TRACKER_HEADS.get("MIN_INSTANCE_PERIOD", 1)
        track_iou_threshold = cfg.TRACKER_HEADS.get("TRACK_IOU_THRESHOLD", 0.5)
        return {
            "_target_": "detectron2.tracking.iou_weighted_hungarian_bbox_iou_tracker.IOUWeightedHungarianBBoxIOUTracker",  # noqa
            "video_height": video_height,
            "video_width": video_width,
            "max_num_instances": max_num_instances,
            "max_lost_frame_count": max_lost_frame_count,
            "min_box_rel_dim": min_box_rel_dim,
            "min_instance_period": min_instance_period,
            "track_iou_threshold": track_iou_threshold,
        }

    def assign_cost_matrix_values(self, cost_matrix: np.ndarray, bbox_pairs: List) -> np.ndarray:
        """
        Based on IoU for each pair of bbox, assign the associated value in cost matrix

        Args:
            cost_matrix: np.ndarray, initialized 2D array with target dimensions
            bbox_pairs: list of bbox pair, in each pair, iou value is stored
        Return:
            np.ndarray, cost_matrix with assigned values
        """
        for pair in bbox_pairs:
            # assign (-1 * IoU) for above threshold pairs, algorithms will minimize cost
            cost_matrix[pair["idx"]][pair["prev_idx"]] = -1 * pair["IoU"]
        return cost_matrix
