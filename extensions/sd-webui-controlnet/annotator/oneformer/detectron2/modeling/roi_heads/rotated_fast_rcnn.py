# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import torch

from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.layers import ShapeSpec, batched_nms_rotated
from annotator.oneformer.detectron2.structures import Instances, RotatedBoxes, pairwise_iou_rotated
from annotator.oneformer.detectron2.utils.events import get_event_storage

from ..box_regression import Box2BoxTransformRotated
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 5-d (dx, dy, dw, dh, da) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransformRotated`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted rotated box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth rotated box2box transform deltas
"""


def fast_rcnn_inference_rotated(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image_rotated` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 5) if doing
            class-specific regression, or (Ri, 5) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_rotated(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


@torch.no_grad()
def fast_rcnn_inference_single_image_rotated(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return rotated bounding-box detection results by thresholding
    on scores and applying rotated non-maximum suppression (Rotated NMS).

    Args:
        Same as `fast_rcnn_inference_rotated`, but with rotated boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference_rotated`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    B = 5  # box dimension
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // B
    # Convert to Boxes to use the `clip` function ...
    boxes = RotatedBoxes(boxes.reshape(-1, B))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, B)  # R x C x B
    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class Rotated NMS
    keep = batched_nms_rotated(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = RotatedBoxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]

    return result, filter_inds[:, 0]


class RotatedFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Two linear layers for predicting Rotated Fast R-CNN outputs.
    """

    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        args["box2box_transform"] = Box2BoxTransformRotated(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )
        return args

    def inference(self, predictions, proposals):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference_rotated`.
            list[Tensor]: same as `fast_rcnn_inference_rotated`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        return fast_rcnn_inference_rotated(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


@ROI_HEADS_REGISTRY.register()
class RROIHeads(StandardROIHeads):
    """
    This class is used by Rotated Fast R-CNN to detect rotated boxes.
    For now, it only supports box predictions but not mask or keypoints.
    """

    @configurable
    def __init__(self, **kwargs):
        """
        NOTE: this interface is experimental.
        """
        super().__init__(**kwargs)
        assert (
            not self.mask_on and not self.keypoint_on
        ), "Mask/Keypoints not supported in Rotated ROIHeads."
        assert not self.train_on_pred_boxes, "train_on_pred_boxes not implemented for RROIHeads!"

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        assert pooler_type in ["ROIAlignRotated"], pooler_type
        # assume all channel counts are equal
        in_channels = [input_shape[f].channels for f in in_features][0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # This line is the only difference v.s. StandardROIHeads
        box_predictor = RotatedFastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the RROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`StandardROIHeads.forward`

        Returns:
            list[Instances]: length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the rotated proposal boxes
                - gt_boxes: the ground-truth rotated boxes that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                - gt_classes: the ground-truth classification lable for each proposal
        """
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou_rotated(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.gt_boxes = targets_per_image.gt_boxes[sampled_targets]

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
