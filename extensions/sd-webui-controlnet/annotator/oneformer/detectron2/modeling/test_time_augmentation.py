# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
from typing import List
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.data.detection_utils import read_image
from annotator.oneformer.detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    apply_augmentations,
)
from annotator.oneformer.detectron2.structures import Boxes, Instances

from .meta_arch import GeneralizedRCNN
from .postprocessing import detector_postprocess
from .roi_heads.fast_rcnn import fast_rcnn_inference_single_image

__all__ = ["DatasetMapperTTA", "GeneralizedRCNNWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    @configurable
    def __init__(self, min_sizes: List[int], max_size: int, flip: bool):
        """
        Args:
            min_sizes: list of short-edge size to resize the image to
            max_size: maximum height or width of resized images
            flip: whether to apply flipping augmentation
        """
        self.min_sizes = min_sizes
        self.max_size = max_size
        self.flip = flip

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            aug_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                aug_candidates.append([resize, flip])  # resize + flip

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class GeneralizedRCNNWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, GeneralizedRCNN
        ), "TTA is only supported on GeneralizedRCNN. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    @contextmanager
    def _turn_off_roi_heads(self, attrs):
        """
        Open a context where some heads in `model.roi_heads` are temporarily turned off.
        Args:
            attr (list[str]): the attribute in `model.roi_heads` which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
        roi_heads = self.model.roi_heads
        old = {}
        for attr in attrs:
            try:
                old[attr] = getattr(roi_heads, attr)
            except AttributeError:
                # The head may not be implemented in certain ROIHeads
                pass

        if len(old.keys()) == 0:
            yield
        else:
            for attr in old.keys():
                setattr(roi_heads, attr, False)
            yield
            for attr in old.keys():
                setattr(roi_heads, attr, old[attr])

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=False,
                    )
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        with self._turn_off_roi_heads(["mask_on", "keypoint_on"]):
            # temporarily disable roi heads
            all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)

        if self.cfg.MODEL.MASK_ON:
            # Use the detected boxes to obtain masks
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, tfms
            )
            # run forward on the detected boxes
            outputs = self._batch_inference(augmented_inputs, augmented_instances)
            # Delete now useless variables to avoid being out of memory
            del augmented_inputs, augmented_instances
            # average the predictions
            merged_instances.pred_masks = self._reduce_pred_masks(outputs, tfms)
            merged_instances = detector_postprocess(merged_instances, *orig_shape)
            return {"instances": merged_instances}
        else:
            return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_boxes(self, augmented_inputs, tfms):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))

            all_scores.extend(output.scores)
            all_classes.extend(output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes, all_scores, all_classes

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        return merged_instances

    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
        augmented_instances = []
        for input, tfm in zip(augmented_inputs, tfms):
            # Transform the target box to the augmented image's coordinate space
            pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
            pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

            aug_instances = Instances(
                image_size=input["image"].shape[1:3],
                pred_boxes=Boxes(pred_boxes),
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        return augmented_instances

    def _reduce_pred_masks(self, outputs, tfms):
        # Should apply inverse transforms on masks.
        # We assume only resize & flip are used. pred_masks is a scale-invariant
        # representation, so we handle flip specially
        for output, tfm in zip(outputs, tfms):
            if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                output.pred_masks = output.pred_masks.flip(dims=[3])
        all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        return avg_pred_masks
