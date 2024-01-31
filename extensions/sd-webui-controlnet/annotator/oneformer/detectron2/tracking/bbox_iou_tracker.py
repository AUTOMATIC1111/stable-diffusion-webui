#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
import copy
import numpy as np
from typing import List
import torch

from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.structures import Boxes, Instances
from annotator.oneformer.detectron2.structures.boxes import pairwise_iou

from ..config.config import CfgNode as CfgNode_
from .base_tracker import TRACKER_HEADS_REGISTRY, BaseTracker


@TRACKER_HEADS_REGISTRY.register()
class BBoxIOUTracker(BaseTracker):
    """
    A bounding box tracker to assign ID based on IoU between current and previous instances
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
        super().__init__(**kwargs)
        self._video_height = video_height
        self._video_width = video_width
        self._max_num_instances = max_num_instances
        self._max_lost_frame_count = max_lost_frame_count
        self._min_box_rel_dim = min_box_rel_dim
        self._min_instance_period = min_instance_period
        self._track_iou_threshold = track_iou_threshold

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
            "_target_": "detectron2.tracking.bbox_iou_tracker.BBoxIOUTracker",
            "video_height": video_height,
            "video_width": video_width,
            "max_num_instances": max_num_instances,
            "max_lost_frame_count": max_lost_frame_count,
            "min_box_rel_dim": min_box_rel_dim,
            "min_instance_period": min_instance_period,
            "track_iou_threshold": track_iou_threshold,
        }

    def update(self, instances: Instances) -> Instances:
        """
        See BaseTracker description
        """
        instances = self._initialize_extra_fields(instances)
        if self._prev_instances is not None:
            # calculate IoU of all bbox pairs
            iou_all = pairwise_iou(
                boxes1=instances.pred_boxes,
                boxes2=self._prev_instances.pred_boxes,
            )
            # sort IoU in descending order
            bbox_pairs = self._create_prediction_pairs(instances, iou_all)
            # assign previous ID to current bbox if IoU > track_iou_threshold
            self._reset_fields()
            for bbox_pair in bbox_pairs:
                idx = bbox_pair["idx"]
                prev_id = bbox_pair["prev_id"]
                if (
                    idx in self._matched_idx
                    or prev_id in self._matched_ID
                    or bbox_pair["IoU"] < self._track_iou_threshold
                ):
                    continue
                instances.ID[idx] = prev_id
                instances.ID_period[idx] = bbox_pair["prev_period"] + 1
                instances.lost_frame_count[idx] = 0
                self._matched_idx.add(idx)
                self._matched_ID.add(prev_id)
                self._untracked_prev_idx.remove(bbox_pair["prev_idx"])
            instances = self._assign_new_id(instances)
            instances = self._merge_untracked_instances(instances)
        self._prev_instances = copy.deepcopy(instances)
        return instances

    def _create_prediction_pairs(self, instances: Instances, iou_all: np.ndarray) -> List:
        """
        For all instances in previous and current frames, create pairs. For each
        pair, store index of the instance in current frame predcitions, index in
        previous predictions, ID in previous predictions, IoU of the bboxes in this
        pair, period in previous predictions.

        Args:
            instances: D2 Instances, for predictions of the current frame
            iou_all: IoU for all bboxes pairs
        Return:
            A list of IoU for all pairs
        """
        bbox_pairs = []
        for i in range(len(instances)):
            for j in range(len(self._prev_instances)):
                bbox_pairs.append(
                    {
                        "idx": i,
                        "prev_idx": j,
                        "prev_id": self._prev_instances.ID[j],
                        "IoU": iou_all[i, j],
                        "prev_period": self._prev_instances.ID_period[j],
                    }
                )
        return bbox_pairs

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

    def _reset_fields(self):
        """
        Before each uodate call, reset fields first
        """
        self._matched_idx = set()
        self._matched_ID = set()
        self._untracked_prev_idx = set(range(len(self._prev_instances)))

    def _assign_new_id(self, instances: Instances) -> Instances:
        """
        For each untracked instance, assign a new id

        Args:
            instances: D2 Instances, for predictions of the current frame
        Return:
            D2 Instances with new ID assigned
        """
        untracked_idx = set(range(len(instances))).difference(self._matched_idx)
        for idx in untracked_idx:
            instances.ID[idx] = self._id_count
            self._id_count += 1
            instances.ID_period[idx] = 1
            instances.lost_frame_count[idx] = 0
        return instances

    def _merge_untracked_instances(self, instances: Instances) -> Instances:
        """
        For untracked previous instances, under certain condition, still keep them
        in tracking and merge with the current instances.

        Args:
            instances: D2 Instances, for predictions of the current frame
        Return:
            D2 Instances merging current instances and instances from previous
            frame decided to keep tracking
        """
        untracked_instances = Instances(
            image_size=instances.image_size,
            pred_boxes=[],
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
            untracked_instances.set("pred_masks", [])
            prev_masks = list(self._prev_instances.pred_masks)
        if instances.has("pred_keypoints"):
            untracked_instances.set("pred_keypoints", [])
            prev_keypoints = list(self._prev_instances.pred_keypoints)
        if instances.has("pred_keypoint_heatmaps"):
            untracked_instances.set("pred_keypoint_heatmaps", [])
            prev_keypoint_heatmaps = list(self._prev_instances.pred_keypoint_heatmaps)
        for idx in self._untracked_prev_idx:
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
            if instances.has("pred_keypoints"):
                untracked_instances.pred_keypoints.append(
                    prev_keypoints[idx].numpy().astype(np.uint8)
                )
            if instances.has("pred_keypoint_heatmaps"):
                untracked_instances.pred_keypoint_heatmaps.append(
                    prev_keypoint_heatmaps[idx].numpy().astype(np.float32)
                )
        untracked_instances.pred_boxes = Boxes(torch.FloatTensor(untracked_instances.pred_boxes))
        untracked_instances.pred_classes = torch.IntTensor(untracked_instances.pred_classes)
        untracked_instances.scores = torch.FloatTensor(untracked_instances.scores)
        if instances.has("pred_masks"):
            untracked_instances.pred_masks = torch.IntTensor(untracked_instances.pred_masks)
        if instances.has("pred_keypoints"):
            untracked_instances.pred_keypoints = torch.IntTensor(untracked_instances.pred_keypoints)
        if instances.has("pred_keypoint_heatmaps"):
            untracked_instances.pred_keypoint_heatmaps = torch.FloatTensor(
                untracked_instances.pred_keypoint_heatmaps
            )

        return Instances.cat(
            [
                instances,
                untracked_instances,
            ]
        )
