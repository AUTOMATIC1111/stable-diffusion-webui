#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import copy
import numpy as np
from typing import Dict
import torch
from scipy.optimize import linear_sum_assignment

from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.structures import Boxes, Instances

from ..config.config import CfgNode as CfgNode_
from .base_tracker import BaseTracker


class BaseHungarianTracker(BaseTracker):
    """
    A base class for all Hungarian trackers
    """

    @configurable
    def __init__(
        self,
        video_height: int,
        video_width: int,
        max_num_instances: int = 200,
        max_lost_frame_count: int = 0,
        min_box_rel_dim: float = 0.02,
        min_instance_period: int = 1,
        **kwargs
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
        """
        super().__init__(**kwargs)
        self._video_height = video_height
        self._video_width = video_width
        self._max_num_instances = max_num_instances
        self._max_lost_frame_count = max_lost_frame_count
        self._min_box_rel_dim = min_box_rel_dim
        self._min_instance_period = min_instance_period

    @classmethod
    def from_config(cls, cfg: CfgNode_) -> Dict:
        raise NotImplementedError("Calling HungarianTracker::from_config")

    def build_cost_matrix(self, instances: Instances, prev_instances: Instances) -> np.ndarray:
        raise NotImplementedError("Calling HungarianTracker::build_matrix")

    def update(self, instances: Instances) -> Instances:
        if instances.has("pred_keypoints"):
            raise NotImplementedError("Need to add support for keypoints")
        instances = self._initialize_extra_fields(instances)
        if self._prev_instances is not None:
            self._untracked_prev_idx = set(range(len(self._prev_instances)))
            cost_matrix = self.build_cost_matrix(instances, self._prev_instances)
            matched_idx, matched_prev_idx = linear_sum_assignment(cost_matrix)
            instances = self._process_matched_idx(instances, matched_idx, matched_prev_idx)
            instances = self._process_unmatched_idx(instances, matched_idx)
            instances = self._process_unmatched_prev_idx(instances, matched_prev_idx)
        self._prev_instances = copy.deepcopy(instances)
        return instances

    def _initialize_extra_fields(self, instances: Instances) -> Instances:
        """
        If input instances don't have ID, ID_period, lost_frame_count fields,
        this method is used to initialize these fields.

        Args:
            instances: D2 Instances, for predictions of the current frame
        Return:
            D2 Instances with extra fields added
        """
        if not instances.has("ID"):
            instances.set("ID", [None] * len(instances))
        if not instances.has("ID_period"):
            instances.set("ID_period", [None] * len(instances))
        if not instances.has("lost_frame_count"):
            instances.set("lost_frame_count", [None] * len(instances))
        if self._prev_instances is None:
            instances.ID = list(range(len(instances)))
            self._id_count += len(instances)
            instances.ID_period = [1] * len(instances)
            instances.lost_frame_count = [0] * len(instances)
        return instances

    def _process_matched_idx(
        self, instances: Instances, matched_idx: np.ndarray, matched_prev_idx: np.ndarray
    ) -> Instances:
        assert matched_idx.size == matched_prev_idx.size
        for i in range(matched_idx.size):
            instances.ID[matched_idx[i]] = self._prev_instances.ID[matched_prev_idx[i]]
            instances.ID_period[matched_idx[i]] = (
                self._prev_instances.ID_period[matched_prev_idx[i]] + 1
            )
            instances.lost_frame_count[matched_idx[i]] = 0
        return instances

    def _process_unmatched_idx(self, instances: Instances, matched_idx: np.ndarray) -> Instances:
        untracked_idx = set(range(len(instances))).difference(set(matched_idx))
        for idx in untracked_idx:
            instances.ID[idx] = self._id_count
            self._id_count += 1
            instances.ID_period[idx] = 1
            instances.lost_frame_count[idx] = 0
        return instances

    def _process_unmatched_prev_idx(
        self, instances: Instances, matched_prev_idx: np.ndarray
    ) -> Instances:
        untracked_instances = Instances(
            image_size=instances.image_size,
            pred_boxes=[],
            pred_masks=[],
            pred_classes=[],
            scores=[],
            ID=[],
            ID_period=[],
            lost_frame_count=[],
        )
        prev_bboxes = list(self._prev_instances.pred_boxes)
        prev_classes = list(self._prev_instances.pred_classes)
        prev_scores = list(self._prev_instances.scores)
        prev_ID_period = self._prev_instances.ID_period
        if instances.has("pred_masks"):
            prev_masks = list(self._prev_instances.pred_masks)
        untracked_prev_idx = set(range(len(self._prev_instances))).difference(set(matched_prev_idx))
        for idx in untracked_prev_idx:
            x_left, y_top, x_right, y_bot = prev_bboxes[idx]
            if (
                (1.0 * (x_right - x_left) / self._video_width < self._min_box_rel_dim)
                or (1.0 * (y_bot - y_top) / self._video_height < self._min_box_rel_dim)
                or self._prev_instances.lost_frame_count[idx] >= self._max_lost_frame_count
                or prev_ID_period[idx] <= self._min_instance_period
            ):
                continue
            untracked_instances.pred_boxes.append(list(prev_bboxes[idx].numpy()))
            untracked_instances.pred_classes.append(int(prev_classes[idx]))
            untracked_instances.scores.append(float(prev_scores[idx]))
            untracked_instances.ID.append(self._prev_instances.ID[idx])
            untracked_instances.ID_period.append(self._prev_instances.ID_period[idx])
            untracked_instances.lost_frame_count.append(
                self._prev_instances.lost_frame_count[idx] + 1
            )
            if instances.has("pred_masks"):
                untracked_instances.pred_masks.append(prev_masks[idx].numpy().astype(np.uint8))

        untracked_instances.pred_boxes = Boxes(torch.FloatTensor(untracked_instances.pred_boxes))
        untracked_instances.pred_classes = torch.IntTensor(untracked_instances.pred_classes)
        untracked_instances.scores = torch.FloatTensor(untracked_instances.scores)
        if instances.has("pred_masks"):
            untracked_instances.pred_masks = torch.IntTensor(untracked_instances.pred_masks)
        else:
            untracked_instances.remove("pred_masks")

        return Instances.cat(
            [
                instances,
                untracked_instances,
            ]
        )
