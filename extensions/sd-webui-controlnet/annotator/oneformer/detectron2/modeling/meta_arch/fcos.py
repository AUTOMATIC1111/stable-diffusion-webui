# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import List, Optional, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F

from annotator.oneformer.detectron2.layers import ShapeSpec, batched_nms
from annotator.oneformer.detectron2.structures import Boxes, ImageList, Instances, pairwise_point_box_distance
from annotator.oneformer.detectron2.utils.events import get_event_storage

from ..anchor_generator import DefaultAnchorGenerator
from ..backbone import Backbone
from ..box_regression import Box2BoxTransformLinear, _dense_box_regression_loss
from .dense_detector import DenseDetector
from .retinanet import RetinaNetHead

__all__ = ["FCOS"]

logger = logging.getLogger(__name__)


class FCOS(DenseDetector):
    """
    Implement FCOS in :paper:`fcos`.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        head_in_features: Optional[List[str]] = None,
        box2box_transform=None,
        num_classes,
        center_sampling_radius: float = 1.5,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        test_score_thresh=0.2,
        test_topk_candidates=1000,
        test_nms_thresh=0.6,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
    ):
        """
        Args:
            center_sampling_radius: radius of the "center" of a groundtruth box,
                within which all anchor points are labeled positive.
            Other arguments mean the same as in :class:`RetinaNet`.
        """
        super().__init__(
            backbone, head, head_in_features, pixel_mean=pixel_mean, pixel_std=pixel_std
        )

        self.num_classes = num_classes

        # FCOS uses one anchor point per location.
        # We represent the anchor point by a box whose size equals the anchor stride.
        feature_shapes = backbone.output_shape()
        fpn_strides = [feature_shapes[k].stride for k in self.head_in_features]
        self.anchor_generator = DefaultAnchorGenerator(
            sizes=[[k] for k in fpn_strides], aspect_ratios=[1.0], strides=fpn_strides
        )

        # FCOS parameterizes box regression by a linear transform,
        # where predictions are normalized by anchor stride (equal to anchor size).
        if box2box_transform is None:
            box2box_transform = Box2BoxTransformLinear(normalize_by_size=True)
        self.box2box_transform = box2box_transform

        self.center_sampling_radius = float(center_sampling_radius)

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image

    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1]
        )
        anchors = self.anchor_generator(features)
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        return self.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centerness
        )

    @torch.no_grad()
    def _match_anchors(self, gt_boxes: Boxes, anchors: List[Boxes]):
        """
        Match ground-truth boxes to a set of multi-level anchors.

        Args:
            gt_boxes: Ground-truth boxes from instances of an image.
            anchors: List of anchors for each feature map (of different scales).

        Returns:
            torch.Tensor
                A tensor of shape `(M, R)`, given `M` ground-truth boxes and total
                `R` anchor points from all feature levels, indicating the quality
                of match between m-th box and r-th anchor. Higher value indicates
                better match.
        """
        # Naming convention: (M = ground-truth boxes, R = anchor points)
        # Anchor points are represented as square boxes of size = stride.
        num_anchors_per_level = [len(x) for x in anchors]
        anchors = Boxes.cat(anchors)  # (R, 4)
        anchor_centers = anchors.get_centers()  # (R, 2)
        anchor_sizes = anchors.tensor[:, 2] - anchors.tensor[:, 0]  # (R, )

        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")

        gt_centers = gt_boxes.get_centers()

        # FCOS with center sampling: anchor point must be close enough to
        # ground-truth box center.
        center_dists = (anchor_centers[None, :, :] - gt_centers[:, None, :]).abs_()
        sampling_regions = self.center_sampling_radius * anchor_sizes[None, :]

        match_quality_matrix = center_dists.max(dim=2).values < sampling_regions

        pairwise_dist = pairwise_point_box_distance(anchor_centers, gt_boxes)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)  # (M, R, 4)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_quality_matrix &= pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        pairwise_dist = pairwise_dist.max(dim=2).values
        match_quality_matrix &= (pairwise_dist > lower_bound[None, :]) & (
            pairwise_dist < upper_bound[None, :]
        )
        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = gt_boxes.area()  # (M, )

        match_quality_matrix = match_quality_matrix.to(torch.float32)
        match_quality_matrix *= 1e8 - gt_areas[:, None]
        return match_quality_matrix  # (M, R)

    @torch.no_grad()
    def label_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        Same interface as :meth:`RetinaNet.label_anchors`, but implemented with FCOS
        anchor matching rule.

        Unlike RetinaNet, there are no ignored anchors.
        """

        gt_labels, matched_gt_boxes = [], []

        for inst in gt_instances:
            if len(inst) > 0:
                match_quality_matrix = self._match_anchors(inst.gt_boxes, anchors)

                # Find matched ground-truth box per anchor. Un-matched anchors are
                # assigned -1. This is equivalent to using an anchor matcher as used
                # in R-CNN/RetinaNet: `Matcher(thresholds=[1e-5], labels=[0, 1])`
                match_quality, matched_idxs = match_quality_matrix.max(dim=0)
                matched_idxs[match_quality < 1e-5] = -1

                matched_gt_boxes_i = inst.gt_boxes.tensor[matched_idxs.clip(min=0)]
                gt_labels_i = inst.gt_classes[matched_idxs.clip(min=0)]

                # Anchors with matched_idxs = -1 are labeled background.
                gt_labels_i[matched_idxs < 0] = self.num_classes
            else:
                matched_gt_boxes_i = torch.zeros_like(Boxes.cat(anchors).tensor)
                gt_labels_i = torch.full(
                    (len(matched_gt_boxes_i),),
                    fill_value=self.num_classes,
                    dtype=torch.long,
                    device=matched_gt_boxes_i.device,
                )

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def losses(
        self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centerness
    ):
        """
        This method is almost identical to :meth:`RetinaNet.losses`, with an extra
        "loss_centerness" in the returned dict.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (M, R)

        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 300)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[
            :, :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            torch.cat(pred_logits, dim=1),
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type="giou",
        )

        ctrness_targets = self.compute_ctrness_targets(anchors, gt_boxes)  # (M, R)
        pred_centerness = torch.cat(pred_centerness, dim=1).squeeze(dim=2)  # (M, R)
        ctrness_loss = F.binary_cross_entropy_with_logits(
            pred_centerness[pos_mask], ctrness_targets[pos_mask], reduction="sum"
        )
        return {
            "loss_fcos_cls": loss_cls / normalizer,
            "loss_fcos_loc": loss_box_reg / normalizer,
            "loss_fcos_ctr": ctrness_loss / normalizer,
        }

    def compute_ctrness_targets(self, anchors: List[Boxes], gt_boxes: List[torch.Tensor]):
        anchors = Boxes.cat(anchors).tensor  # Rx4
        reg_targets = [self.box2box_transform.get_deltas(anchors, m) for m in gt_boxes]
        reg_targets = torch.stack(reg_targets, dim=0)  # NxRx4
        if len(reg_targets) == 0:
            return reg_targets.new_zeros(len(reg_targets))
        left_right = reg_targets[:, :, [0, 2]]
        top_bottom = reg_targets[:, :, [1, 3]]
        ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
        )
        return torch.sqrt(ctrness)

    def forward_inference(
        self,
        images: ImageList,
        features: List[torch.Tensor],
        predictions: List[List[torch.Tensor]],
    ):
        pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4, 1]
        )
        anchors = self.anchor_generator(features)

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [
                # Multiply and sqrt centerness & classification scores
                # (See eqn. 4 in https://arxiv.org/abs/2006.09214)
                torch.sqrt(x[img_idx].sigmoid_() * y[img_idx].sigmoid_())
                for x, y in zip(pred_logits, pred_centerness)
            ]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        anchors: List[Boxes],
        box_cls: List[torch.Tensor],
        box_delta: List[torch.Tensor],
        image_size: Tuple[int, int],
    ):
        """
        Identical to :meth:`RetinaNet.inference_single_image.
        """
        pred = self._decode_multi_level_predictions(
            anchors,
            box_cls,
            box_delta,
            self.test_score_thresh,
            self.test_topk_candidates,
            image_size,
        )
        keep = batched_nms(
            pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh
        )
        return pred[keep[: self.max_detections_per_image]]


class FCOSHead(RetinaNetHead):
    """
    The head used in :paper:`fcos`. It adds an additional centerness
    prediction branch on top of :class:`RetinaNetHead`.
    """

    def __init__(self, *, input_shape: List[ShapeSpec], conv_dims: List[int], **kwargs):
        super().__init__(input_shape=input_shape, conv_dims=conv_dims, num_anchors=1, **kwargs)
        # Unlike original FCOS, we do not add an additional learnable scale layer
        # because it's found to have no benefits after normalizing regression targets by stride.
        self._num_features = len(input_shape)
        self.ctrness = nn.Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.ctrness.weight, std=0.01)
        torch.nn.init.constant_(self.ctrness.bias, 0)

    def forward(self, features):
        assert len(features) == self._num_features
        logits = []
        bbox_reg = []
        ctrness = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_feature = self.bbox_subnet(feature)
            bbox_reg.append(self.bbox_pred(bbox_feature))
            ctrness.append(self.ctrness(bbox_feature))
        return logits, bbox_reg, ctrness
