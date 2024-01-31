# Copyright (c) Facebook, Inc. and its affiliates.
from .boxes import Boxes, BoxMode, pairwise_iou, pairwise_ioa, pairwise_point_box_distance
from .image_list import ImageList

from .instances import Instances
from .keypoints import Keypoints, heatmaps_to_keypoints
from .masks import BitMasks, PolygonMasks, polygons_to_bitmask, ROIMasks
from .rotated_boxes import RotatedBoxes
from .rotated_boxes import pairwise_iou as pairwise_iou_rotated

__all__ = [k for k in globals().keys() if not k.startswith("_")]


from annotator.oneformer.detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
