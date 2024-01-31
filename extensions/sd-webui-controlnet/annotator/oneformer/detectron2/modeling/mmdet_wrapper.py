# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Union
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from annotator.oneformer.detectron2.layers import ShapeSpec
from annotator.oneformer.detectron2.structures import BitMasks, Boxes, ImageList, Instances
from annotator.oneformer.detectron2.utils.events import get_event_storage

from .backbone import Backbone

logger = logging.getLogger(__name__)


def _to_container(cfg):
    """
    mmdet will assert the type of dict/list.
    So convert omegaconf objects to dict/list.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    from mmcv.utils import ConfigDict

    return ConfigDict(cfg)


class MMDetBackbone(Backbone):
    """
    Wrapper of mmdetection backbones to use in detectron2.

    mmdet backbones produce list/tuple of tensors, while detectron2 backbones
    produce a dict of tensors. This class wraps the given backbone to produce
    output in detectron2's convention, so it can be used in place of detectron2
    backbones.
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Mapping],
        neck: Union[nn.Module, Mapping, None] = None,
        *,
        output_shapes: List[ShapeSpec],
        output_names: Optional[List[str]] = None,
    ):
        """
        Args:
            backbone: either a backbone module or a mmdet config dict that defines a
                backbone. The backbone takes a 4D image tensor and returns a
                sequence of tensors.
            neck: either a backbone module or a mmdet config dict that defines a
                neck. The neck takes outputs of backbone and returns a
                sequence of tensors. If None, no neck is used.
            output_shapes: shape for every output of the backbone (or neck, if given).
                stride and channels are often needed.
            output_names: names for every output of the backbone (or neck, if given).
                By default, will use "out0", "out1", ...
        """
        super().__init__()
        if isinstance(backbone, Mapping):
            from mmdet.models import build_backbone

            backbone = build_backbone(_to_container(backbone))
        self.backbone = backbone

        if isinstance(neck, Mapping):
            from mmdet.models import build_neck

            neck = build_neck(_to_container(neck))
        self.neck = neck

        # "Neck" weights, if any, are part of neck itself. This is the interface
        # of mmdet so we follow it. Reference:
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/two_stage.py
        logger.info("Initializing mmdet backbone weights...")
        self.backbone.init_weights()
        # train() in mmdet modules is non-trivial, and has to be explicitly
        # called. Reference:
        # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py
        self.backbone.train()
        if self.neck is not None:
            logger.info("Initializing mmdet neck weights ...")
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
            self.neck.train()

        self._output_shapes = output_shapes
        if not output_names:
            output_names = [f"out{i}" for i in range(len(output_shapes))]
        self._output_names = output_names

    def forward(self, x) -> Dict[str, Tensor]:
        outs = self.backbone(x)
        if self.neck is not None:
            outs = self.neck(outs)
        assert isinstance(
            outs, (list, tuple)
        ), "mmdet backbone should return a list/tuple of tensors!"
        if len(outs) != len(self._output_shapes):
            raise ValueError(
                "Length of output_shapes does not match outputs from the mmdet backbone: "
                f"{len(outs)} != {len(self._output_shapes)}"
            )
        return {k: v for k, v in zip(self._output_names, outs)}

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {k: v for k, v in zip(self._output_names, self._output_shapes)}


class MMDetDetector(nn.Module):
    """
    Wrapper of a mmdetection detector model, for detection and instance segmentation.
    Input/output formats of this class follow detectron2's convention, so a
    mmdetection model can be trained and evaluated in detectron2.
    """

    def __init__(
        self,
        detector: Union[nn.Module, Mapping],
        *,
        # Default is 32 regardless of model:
        # https://github.com/open-mmlab/mmdetection/tree/master/configs/_base_/datasets
        size_divisibility=32,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            detector: a mmdet detector, or a mmdet config dict that defines a detector.
            size_divisibility: pad input images to multiple of this number
            pixel_mean: per-channel mean to normalize input image
            pixel_std: per-channel stddev to normalize input image
        """
        super().__init__()
        if isinstance(detector, Mapping):
            from mmdet.models import build_detector

            detector = build_detector(_to_container(detector))
        self.detector = detector
        self.detector.init_weights()
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility=self.size_divisibility).tensor
        metas = []
        rescale = {"height" in x for x in batched_inputs}
        if len(rescale) != 1:
            raise ValueError("Some inputs have original height/width, but some don't!")
        rescale = list(rescale)[0]
        output_shapes = []
        for input in batched_inputs:
            meta = {}
            c, h, w = input["image"].shape
            meta["img_shape"] = meta["ori_shape"] = (h, w, c)
            if rescale:
                scale_factor = np.array(
                    [w / input["width"], h / input["height"]] * 2, dtype="float32"
                )
                ori_shape = (input["height"], input["width"])
                output_shapes.append(ori_shape)
                meta["ori_shape"] = ori_shape + (c,)
            else:
                scale_factor = 1.0
                output_shapes.append((h, w))
            meta["scale_factor"] = scale_factor
            meta["flip"] = False
            padh, padw = images.shape[-2:]
            meta["pad_shape"] = (padh, padw, c)
            metas.append(meta)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if gt_instances[0].has("gt_masks"):
                from mmdet.core import PolygonMasks as mm_PolygonMasks, BitmapMasks as mm_BitMasks

                def convert_mask(m, shape):
                    # mmdet mask format
                    if isinstance(m, BitMasks):
                        return mm_BitMasks(m.tensor.cpu().numpy(), shape[0], shape[1])
                    else:
                        return mm_PolygonMasks(m.polygons, shape[0], shape[1])

                gt_masks = [convert_mask(x.gt_masks, x.image_size) for x in gt_instances]
                losses_and_metrics = self.detector.forward_train(
                    images,
                    metas,
                    [x.gt_boxes.tensor for x in gt_instances],
                    [x.gt_classes for x in gt_instances],
                    gt_masks=gt_masks,
                )
            else:
                losses_and_metrics = self.detector.forward_train(
                    images,
                    metas,
                    [x.gt_boxes.tensor for x in gt_instances],
                    [x.gt_classes for x in gt_instances],
                )
            return _parse_losses(losses_and_metrics)
        else:
            results = self.detector.simple_test(images, metas, rescale=rescale)
            results = [
                {"instances": _convert_mmdet_result(r, shape)}
                for r, shape in zip(results, output_shapes)
            ]
            return results

    @property
    def device(self):
        return self.pixel_mean.device


# Reference: show_result() in
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/base.py
def _convert_mmdet_result(result, shape: Tuple[int, int]) -> Instances:
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None

    bboxes = torch.from_numpy(np.vstack(bbox_result))  # Nx5
    bboxes, scores = bboxes[:, :4], bboxes[:, -1]
    labels = [
        torch.full((bbox.shape[0],), i, dtype=torch.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = torch.cat(labels)
    inst = Instances(shape)
    inst.pred_boxes = Boxes(bboxes)
    inst.scores = scores
    inst.pred_classes = labels

    if segm_result is not None and len(labels) > 0:
        segm_result = list(itertools.chain(*segm_result))
        segm_result = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in segm_result]
        segm_result = torch.stack(segm_result, dim=0)
        inst.pred_masks = segm_result
    return inst


# reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/base.py
def _parse_losses(losses: Dict[str, Tensor]) -> Dict[str, Tensor]:
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        if "loss" not in loss_name:
            # put metrics to storage; don't return them
            storage = get_event_storage()
            value = log_vars.pop(loss_name).cpu().item()
            storage.put_scalar(loss_name, value)
    return log_vars
