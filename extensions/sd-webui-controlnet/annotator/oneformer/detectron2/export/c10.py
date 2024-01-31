# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Dict
import torch
import torch.nn.functional as F

from annotator.oneformer.detectron2.layers import ShapeSpec, cat
from annotator.oneformer.detectron2.layers.roi_align_rotated import ROIAlignRotated
from annotator.oneformer.detectron2.modeling import poolers
from annotator.oneformer.detectron2.modeling.proposal_generator import rpn
from annotator.oneformer.detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from annotator.oneformer.detectron2.structures import Boxes, ImageList, Instances, Keypoints, RotatedBoxes

from .shared import alias, to_device


"""
This file contains caffe2-compatible implementation of several detectron2 components.
"""


class Caffe2Boxes(Boxes):
    """
    Representing a list of detectron2.structures.Boxes from minibatch, each box
    is represented by a 5d vector (batch index + 4 coordinates), or a 6d vector
    (batch index + 5 coordinates) for RotatedBoxes.
    """

    def __init__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 2 and tensor.size(-1) in [4, 5, 6], tensor.size()
        # TODO: make tensor immutable when dim is Nx5 for Boxes,
        # and Nx6 for RotatedBoxes?
        self.tensor = tensor


# TODO clean up this class, maybe just extend Instances
class InstancesList(object):
    """
    Tensor representation of a list of Instances object for a batch of images.

    When dealing with a batch of images with Caffe2 ops, a list of bboxes
    (instances) are usually represented by single Tensor with size
    (sigma(Ni), 5) or (sigma(Ni), 4) plus a batch split Tensor. This class is
    for providing common functions to convert between these two representations.
    """

    def __init__(self, im_info, indices, extra_fields=None):
        # [N, 3] -> (H, W, Scale)
        self.im_info = im_info
        # [N,] -> indice of batch to which the instance belongs
        self.indices = indices
        # [N, ...]
        self.batch_extra_fields = extra_fields or {}

        self.image_size = self.im_info

    def get_fields(self):
        """like `get_fields` in the Instances object,
        but return each field in tensor representations"""
        ret = {}
        for k, v in self.batch_extra_fields.items():
            # if isinstance(v, torch.Tensor):
            #     tensor_rep = v
            # elif isinstance(v, (Boxes, Keypoints)):
            #     tensor_rep = v.tensor
            # else:
            #     raise ValueError("Can't find tensor representation for: {}".format())
            ret[k] = v
        return ret

    def has(self, name):
        return name in self.batch_extra_fields

    def set(self, name, value):
        # len(tensor) is a bad practice that generates ONNX constants during tracing.
        # Although not a problem for the `assert` statement below, torch ONNX exporter
        # still raises a misleading warning as it does not this call comes from `assert`
        if isinstance(value, Boxes):
            data_len = value.tensor.shape[0]
        elif isinstance(value, torch.Tensor):
            data_len = value.shape[0]
        else:
            data_len = len(value)
        if len(self.batch_extra_fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self.batch_extra_fields[name] = value

    def __getattr__(self, name):
        if name not in self.batch_extra_fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self.batch_extra_fields[name]

    def __len__(self):
        return len(self.indices)

    def flatten(self):
        ret = []
        for _, v in self.batch_extra_fields.items():
            if isinstance(v, (Boxes, Keypoints)):
                ret.append(v.tensor)
            else:
                ret.append(v)
        return ret

    @staticmethod
    def to_d2_instances_list(instances_list):
        """
        Convert InstancesList to List[Instances]. The input `instances_list` can
        also be a List[Instances], in this case this method is a non-op.
        """
        if not isinstance(instances_list, InstancesList):
            assert all(isinstance(x, Instances) for x in instances_list)
            return instances_list

        ret = []
        for i, info in enumerate(instances_list.im_info):
            instances = Instances(torch.Size([int(info[0].item()), int(info[1].item())]))

            ids = instances_list.indices == i
            for k, v in instances_list.batch_extra_fields.items():
                if isinstance(v, torch.Tensor):
                    instances.set(k, v[ids])
                    continue
                elif isinstance(v, Boxes):
                    instances.set(k, v[ids, -4:])
                    continue

                target_type, tensor_source = v
                assert isinstance(tensor_source, torch.Tensor)
                assert tensor_source.shape[0] == instances_list.indices.shape[0]
                tensor_source = tensor_source[ids]

                if issubclass(target_type, Boxes):
                    instances.set(k, Boxes(tensor_source[:, -4:]))
                elif issubclass(target_type, Keypoints):
                    instances.set(k, Keypoints(tensor_source))
                elif issubclass(target_type, torch.Tensor):
                    instances.set(k, tensor_source)
                else:
                    raise ValueError("Can't handle targe type: {}".format(target_type))

            ret.append(instances)
        return ret


class Caffe2Compatible(object):
    """
    A model can inherit this class to indicate that it can be traced and deployed with caffe2.
    """

    def _get_tensor_mode(self):
        return self._tensor_mode

    def _set_tensor_mode(self, v):
        self._tensor_mode = v

    tensor_mode = property(_get_tensor_mode, _set_tensor_mode)
    """
    If true, the model expects C2-style tensor only inputs/outputs format.
    """


class Caffe2RPN(Caffe2Compatible, rpn.RPN):
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super(Caffe2Compatible, cls).from_config(cfg, input_shape)
        assert tuple(cfg.MODEL.RPN.BBOX_REG_WEIGHTS) == (1.0, 1.0, 1.0, 1.0) or tuple(
            cfg.MODEL.RPN.BBOX_REG_WEIGHTS
        ) == (1.0, 1.0, 1.0, 1.0, 1.0)
        return ret

    def _generate_proposals(
        self, images, objectness_logits_pred, anchor_deltas_pred, gt_instances=None
    ):
        assert isinstance(images, ImageList)
        if self.tensor_mode:
            im_info = images.image_sizes
        else:
            im_info = torch.tensor([[im_sz[0], im_sz[1], 1.0] for im_sz in images.image_sizes]).to(
                images.tensor.device
            )
        assert isinstance(im_info, torch.Tensor)

        rpn_rois_list = []
        rpn_roi_probs_list = []
        for scores, bbox_deltas, cell_anchors_tensor, feat_stride in zip(
            objectness_logits_pred,
            anchor_deltas_pred,
            [b for (n, b) in self.anchor_generator.cell_anchors.named_buffers()],
            self.anchor_generator.strides,
        ):
            scores = scores.detach()
            bbox_deltas = bbox_deltas.detach()

            rpn_rois, rpn_roi_probs = torch.ops._caffe2.GenerateProposals(
                scores,
                bbox_deltas,
                im_info,
                cell_anchors_tensor,
                spatial_scale=1.0 / feat_stride,
                pre_nms_topN=self.pre_nms_topk[self.training],
                post_nms_topN=self.post_nms_topk[self.training],
                nms_thresh=self.nms_thresh,
                min_size=self.min_box_size,
                # correct_transform_coords=True,  # deprecated argument
                angle_bound_on=True,  # Default
                angle_bound_lo=-180,
                angle_bound_hi=180,
                clip_angle_thresh=1.0,  # Default
                legacy_plus_one=False,
            )
            rpn_rois_list.append(rpn_rois)
            rpn_roi_probs_list.append(rpn_roi_probs)

        # For FPN in D2, in RPN all proposals from different levels are concated
        # together, ranked and picked by top post_nms_topk. Then in ROIPooler
        # it calculates level_assignments and calls the RoIAlign from
        # the corresponding level.

        if len(objectness_logits_pred) == 1:
            rpn_rois = rpn_rois_list[0]
            rpn_roi_probs = rpn_roi_probs_list[0]
        else:
            assert len(rpn_rois_list) == len(rpn_roi_probs_list)
            rpn_post_nms_topN = self.post_nms_topk[self.training]

            device = rpn_rois_list[0].device
            input_list = [to_device(x, "cpu") for x in (rpn_rois_list + rpn_roi_probs_list)]

            # TODO remove this after confirming rpn_max_level/rpn_min_level
            # is not needed in CollectRpnProposals.
            feature_strides = list(self.anchor_generator.strides)
            rpn_min_level = int(math.log2(feature_strides[0]))
            rpn_max_level = int(math.log2(feature_strides[-1]))
            assert (rpn_max_level - rpn_min_level + 1) == len(
                rpn_rois_list
            ), "CollectRpnProposals requires continuous levels"

            rpn_rois = torch.ops._caffe2.CollectRpnProposals(
                input_list,
                # NOTE: in current implementation, rpn_max_level and rpn_min_level
                # are not needed, only the subtraction of two matters and it
                # can be infer from the number of inputs. Keep them now for
                # consistency.
                rpn_max_level=2 + len(rpn_rois_list) - 1,
                rpn_min_level=2,
                rpn_post_nms_topN=rpn_post_nms_topN,
            )
            rpn_rois = to_device(rpn_rois, device)
            rpn_roi_probs = []

        proposals = self.c2_postprocess(im_info, rpn_rois, rpn_roi_probs, self.tensor_mode)
        return proposals, {}

    def forward(self, images, features, gt_instances=None):
        assert not self.training
        features = [features[f] for f in self.in_features]
        objectness_logits_pred, anchor_deltas_pred = self.rpn_head(features)
        return self._generate_proposals(
            images,
            objectness_logits_pred,
            anchor_deltas_pred,
            gt_instances,
        )

    @staticmethod
    def c2_postprocess(im_info, rpn_rois, rpn_roi_probs, tensor_mode):
        proposals = InstancesList(
            im_info=im_info,
            indices=rpn_rois[:, 0],
            extra_fields={
                "proposal_boxes": Caffe2Boxes(rpn_rois),
                "objectness_logits": (torch.Tensor, rpn_roi_probs),
            },
        )
        if not tensor_mode:
            proposals = InstancesList.to_d2_instances_list(proposals)
        else:
            proposals = [proposals]
        return proposals


class Caffe2ROIPooler(Caffe2Compatible, poolers.ROIPooler):
    @staticmethod
    def c2_preprocess(box_lists):
        assert all(isinstance(x, Boxes) for x in box_lists)
        if all(isinstance(x, Caffe2Boxes) for x in box_lists):
            # input is pure-tensor based
            assert len(box_lists) == 1
            pooler_fmt_boxes = box_lists[0].tensor
        else:
            pooler_fmt_boxes = poolers.convert_boxes_to_pooler_format(box_lists)
        return pooler_fmt_boxes

    def forward(self, x, box_lists):
        assert not self.training

        pooler_fmt_boxes = self.c2_preprocess(box_lists)
        num_level_assignments = len(self.level_poolers)

        if num_level_assignments == 1:
            if isinstance(self.level_poolers[0], ROIAlignRotated):
                c2_roi_align = torch.ops._caffe2.RoIAlignRotated
                aligned = True
            else:
                c2_roi_align = torch.ops._caffe2.RoIAlign
                aligned = self.level_poolers[0].aligned

            x0 = x[0]
            if x0.is_quantized:
                x0 = x0.dequantize()

            out = c2_roi_align(
                x0,
                pooler_fmt_boxes,
                order="NCHW",
                spatial_scale=float(self.level_poolers[0].spatial_scale),
                pooled_h=int(self.output_size[0]),
                pooled_w=int(self.output_size[1]),
                sampling_ratio=int(self.level_poolers[0].sampling_ratio),
                aligned=aligned,
            )
            return out

        device = pooler_fmt_boxes.device
        assert (
            self.max_level - self.min_level + 1 == 4
        ), "Currently DistributeFpnProposals only support 4 levels"
        fpn_outputs = torch.ops._caffe2.DistributeFpnProposals(
            to_device(pooler_fmt_boxes, "cpu"),
            roi_canonical_scale=self.canonical_box_size,
            roi_canonical_level=self.canonical_level,
            roi_max_level=self.max_level,
            roi_min_level=self.min_level,
            legacy_plus_one=False,
        )
        fpn_outputs = [to_device(x, device) for x in fpn_outputs]

        rois_fpn_list = fpn_outputs[:-1]
        rois_idx_restore_int32 = fpn_outputs[-1]

        roi_feat_fpn_list = []
        for roi_fpn, x_level, pooler in zip(rois_fpn_list, x, self.level_poolers):
            if isinstance(pooler, ROIAlignRotated):
                c2_roi_align = torch.ops._caffe2.RoIAlignRotated
                aligned = True
            else:
                c2_roi_align = torch.ops._caffe2.RoIAlign
                aligned = bool(pooler.aligned)

            if x_level.is_quantized:
                x_level = x_level.dequantize()

            roi_feat_fpn = c2_roi_align(
                x_level,
                roi_fpn,
                order="NCHW",
                spatial_scale=float(pooler.spatial_scale),
                pooled_h=int(self.output_size[0]),
                pooled_w=int(self.output_size[1]),
                sampling_ratio=int(pooler.sampling_ratio),
                aligned=aligned,
            )
            roi_feat_fpn_list.append(roi_feat_fpn)

        roi_feat_shuffled = cat(roi_feat_fpn_list, dim=0)
        assert roi_feat_shuffled.numel() > 0 and rois_idx_restore_int32.numel() > 0, (
            "Caffe2 export requires tracing with a model checkpoint + input that can produce valid"
            " detections. But no detections were obtained with the given checkpoint and input!"
        )
        roi_feat = torch.ops._caffe2.BatchPermutation(roi_feat_shuffled, rois_idx_restore_int32)
        return roi_feat


class Caffe2FastRCNNOutputsInference:
    def __init__(self, tensor_mode):
        self.tensor_mode = tensor_mode  # whether the output is caffe2 tensor mode

    def __call__(self, box_predictor, predictions, proposals):
        """equivalent to FastRCNNOutputLayers.inference"""
        num_classes = box_predictor.num_classes
        score_thresh = box_predictor.test_score_thresh
        nms_thresh = box_predictor.test_nms_thresh
        topk_per_image = box_predictor.test_topk_per_image
        is_rotated = len(box_predictor.box2box_transform.weights) == 5

        if is_rotated:
            box_dim = 5
            assert box_predictor.box2box_transform.weights[4] == 1, (
                "The weights for Rotated BBoxTransform in C2 have only 4 dimensions,"
                + " thus enforcing the angle weight to be 1 for now"
            )
            box2box_transform_weights = box_predictor.box2box_transform.weights[:4]
        else:
            box_dim = 4
            box2box_transform_weights = box_predictor.box2box_transform.weights

        class_logits, box_regression = predictions
        if num_classes + 1 == class_logits.shape[1]:
            class_prob = F.softmax(class_logits, -1)
        else:
            assert num_classes == class_logits.shape[1]
            class_prob = F.sigmoid(class_logits)
            # BoxWithNMSLimit will infer num_classes from the shape of the class_prob
            # So append a zero column as placeholder for the background class
            class_prob = torch.cat((class_prob, torch.zeros(class_prob.shape[0], 1)), dim=1)

        assert box_regression.shape[1] % box_dim == 0
        cls_agnostic_bbox_reg = box_regression.shape[1] // box_dim == 1

        input_tensor_mode = proposals[0].proposal_boxes.tensor.shape[1] == box_dim + 1

        proposal_boxes = proposals[0].proposal_boxes
        if isinstance(proposal_boxes, Caffe2Boxes):
            rois = Caffe2Boxes.cat([p.proposal_boxes for p in proposals])
        elif isinstance(proposal_boxes, RotatedBoxes):
            rois = RotatedBoxes.cat([p.proposal_boxes for p in proposals])
        elif isinstance(proposal_boxes, Boxes):
            rois = Boxes.cat([p.proposal_boxes for p in proposals])
        else:
            raise NotImplementedError(
                'Expected proposals[0].proposal_boxes to be type "Boxes", '
                f"instead got {type(proposal_boxes)}"
            )

        device, dtype = rois.tensor.device, rois.tensor.dtype
        if input_tensor_mode:
            im_info = proposals[0].image_size
            rois = rois.tensor
        else:
            im_info = torch.tensor(
                [[sz[0], sz[1], 1.0] for sz in [x.image_size for x in proposals]]
            )
            batch_ids = cat(
                [
                    torch.full((b, 1), i, dtype=dtype, device=device)
                    for i, b in enumerate(len(p) for p in proposals)
                ],
                dim=0,
            )
            rois = torch.cat([batch_ids, rois.tensor], dim=1)

        roi_pred_bbox, roi_batch_splits = torch.ops._caffe2.BBoxTransform(
            to_device(rois, "cpu"),
            to_device(box_regression, "cpu"),
            to_device(im_info, "cpu"),
            weights=box2box_transform_weights,
            apply_scale=True,
            rotated=is_rotated,
            angle_bound_on=True,
            angle_bound_lo=-180,
            angle_bound_hi=180,
            clip_angle_thresh=1.0,
            legacy_plus_one=False,
        )
        roi_pred_bbox = to_device(roi_pred_bbox, device)
        roi_batch_splits = to_device(roi_batch_splits, device)

        nms_outputs = torch.ops._caffe2.BoxWithNMSLimit(
            to_device(class_prob, "cpu"),
            to_device(roi_pred_bbox, "cpu"),
            to_device(roi_batch_splits, "cpu"),
            score_thresh=float(score_thresh),
            nms=float(nms_thresh),
            detections_per_im=int(topk_per_image),
            soft_nms_enabled=False,
            soft_nms_method="linear",
            soft_nms_sigma=0.5,
            soft_nms_min_score_thres=0.001,
            rotated=is_rotated,
            cls_agnostic_bbox_reg=cls_agnostic_bbox_reg,
            input_boxes_include_bg_cls=False,
            output_classes_include_bg_cls=False,
            legacy_plus_one=False,
        )
        roi_score_nms = to_device(nms_outputs[0], device)
        roi_bbox_nms = to_device(nms_outputs[1], device)
        roi_class_nms = to_device(nms_outputs[2], device)
        roi_batch_splits_nms = to_device(nms_outputs[3], device)
        roi_keeps_nms = to_device(nms_outputs[4], device)
        roi_keeps_size_nms = to_device(nms_outputs[5], device)
        if not self.tensor_mode:
            roi_class_nms = roi_class_nms.to(torch.int64)

        roi_batch_ids = cat(
            [
                torch.full((b, 1), i, dtype=dtype, device=device)
                for i, b in enumerate(int(x.item()) for x in roi_batch_splits_nms)
            ],
            dim=0,
        )

        roi_class_nms = alias(roi_class_nms, "class_nms")
        roi_score_nms = alias(roi_score_nms, "score_nms")
        roi_bbox_nms = alias(roi_bbox_nms, "bbox_nms")
        roi_batch_splits_nms = alias(roi_batch_splits_nms, "batch_splits_nms")
        roi_keeps_nms = alias(roi_keeps_nms, "keeps_nms")
        roi_keeps_size_nms = alias(roi_keeps_size_nms, "keeps_size_nms")

        results = InstancesList(
            im_info=im_info,
            indices=roi_batch_ids[:, 0],
            extra_fields={
                "pred_boxes": Caffe2Boxes(roi_bbox_nms),
                "scores": roi_score_nms,
                "pred_classes": roi_class_nms,
            },
        )

        if not self.tensor_mode:
            results = InstancesList.to_d2_instances_list(results)
            batch_splits = roi_batch_splits_nms.int().tolist()
            kept_indices = list(roi_keeps_nms.to(torch.int64).split(batch_splits))
        else:
            results = [results]
            kept_indices = [roi_keeps_nms]

        return results, kept_indices


class Caffe2MaskRCNNInference:
    def __call__(self, pred_mask_logits, pred_instances):
        """equivalent to mask_head.mask_rcnn_inference"""
        if all(isinstance(x, InstancesList) for x in pred_instances):
            assert len(pred_instances) == 1
            mask_probs_pred = pred_mask_logits.sigmoid()
            mask_probs_pred = alias(mask_probs_pred, "mask_fcn_probs")
            pred_instances[0].set("pred_masks", mask_probs_pred)
        else:
            mask_rcnn_inference(pred_mask_logits, pred_instances)


class Caffe2KeypointRCNNInference:
    def __init__(self, use_heatmap_max_keypoint):
        self.use_heatmap_max_keypoint = use_heatmap_max_keypoint

    def __call__(self, pred_keypoint_logits, pred_instances):
        # just return the keypoint heatmap for now,
        # there will be option to call HeatmapMaxKeypointOp
        output = alias(pred_keypoint_logits, "kps_score")
        if all(isinstance(x, InstancesList) for x in pred_instances):
            assert len(pred_instances) == 1
            if self.use_heatmap_max_keypoint:
                device = output.device
                output = torch.ops._caffe2.HeatmapMaxKeypoint(
                    to_device(output, "cpu"),
                    pred_instances[0].pred_boxes.tensor,
                    should_output_softmax=True,  # worth make it configerable?
                )
                output = to_device(output, device)
                output = alias(output, "keypoints_out")
            pred_instances[0].set("pred_keypoints", output)
        return pred_keypoint_logits
