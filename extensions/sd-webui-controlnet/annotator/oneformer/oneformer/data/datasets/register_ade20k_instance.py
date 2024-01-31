# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_instance.py
# ------------------------------------------------------------------------------

import json
import logging
import numpy as np
import os
from PIL import Image

from annotator.oneformer.detectron2.data import DatasetCatalog, MetadataCatalog
from annotator.oneformer.detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from annotator.oneformer.detectron2.utils.file_io import PathManager

ADE_CATEGORIES = [{'id': 7, 'name': 'bed'}, {'id': 8, 'name': 'windowpane'}, {'id': 10, 'name': 'cabinet'}, {'id': 12, 'name': 'person'}, {'id': 14, 'name': 'door'}, {'id': 15, 'name': 'table'}, {'id': 18, 'name': 'curtain'}, {'id': 19, 'name': 'chair'}, {'id': 20, 'name': 'car'}, {'id': 22, 'name': 'painting'}, {'id': 23, 'name': 'sofa'}, {'id': 24, 'name': 'shelf'}, {'id': 27, 'name': 'mirror'}, {'id': 30, 'name': 'armchair'}, {'id': 31, 'name': 'seat'}, {'id': 32, 'name': 'fence'}, {'id': 33, 'name': 'desk'}, {'id': 35, 'name': 'wardrobe'}, {'id': 36, 'name': 'lamp'}, {'id': 37, 'name': 'bathtub'}, {'id': 38, 'name': 'railing'}, {'id': 39, 'name': 'cushion'}, {'id': 41, 'name': 'box'}, {'id': 42, 'name': 'column'}, {'id': 43, 'name': 'signboard'}, {'id': 44, 'name': 'chest of drawers'}, {'id': 45, 'name': 'counter'}, {'id': 47, 'name': 'sink'}, {'id': 49, 'name': 'fireplace'}, {'id': 50, 'name': 'refrigerator'}, {'id': 53, 'name': 'stairs'}, {'id': 55, 'name': 'case'}, {'id': 56, 'name': 'pool table'}, {'id': 57, 'name': 'pillow'}, {'id': 58, 'name': 'screen door'}, {'id': 62, 'name': 'bookcase'}, {'id': 64, 'name': 'coffee table'}, {'id': 65, 'name': 'toilet'}, {'id': 66, 'name': 'flower'}, {'id': 67, 'name': 'book'}, {'id': 69, 'name': 'bench'}, {'id': 70, 'name': 'countertop'}, {'id': 71, 'name': 'stove'}, {'id': 72, 'name': 'palm'}, {'id': 73, 'name': 'kitchen island'}, {'id': 74, 'name': 'computer'}, {'id': 75, 'name': 'swivel chair'}, {'id': 76, 'name': 'boat'}, {'id': 78, 'name': 'arcade machine'}, {'id': 80, 'name': 'bus'}, {'id': 81, 'name': 'towel'}, {'id': 82, 'name': 'light'}, {'id': 83, 'name': 'truck'}, {'id': 85, 'name': 'chandelier'}, {'id': 86, 'name': 'awning'}, {'id': 87, 'name': 'streetlight'}, {'id': 88, 'name': 'booth'}, {'id': 89, 'name': 'television receiver'}, {'id': 90, 'name': 'airplane'}, {'id': 92, 'name': 'apparel'}, {'id': 93, 'name': 'pole'}, {'id': 95, 'name': 'bannister'}, {'id': 97, 'name': 'ottoman'}, {'id': 98, 'name': 'bottle'}, {'id': 102, 'name': 'van'}, {'id': 103, 'name': 'ship'}, {'id': 104, 'name': 'fountain'}, {'id': 107, 'name': 'washer'}, {'id': 108, 'name': 'plaything'}, {'id': 110, 'name': 'stool'}, {'id': 111, 'name': 'barrel'}, {'id': 112, 'name': 'basket'}, {'id': 115, 'name': 'bag'}, {'id': 116, 'name': 'minibike'}, {'id': 118, 'name': 'oven'}, {'id': 119, 'name': 'ball'}, {'id': 120, 'name': 'food'}, {'id': 121, 'name': 'step'}, {'id': 123, 'name': 'trade name'}, {'id': 124, 'name': 'microwave'}, {'id': 125, 'name': 'pot'}, {'id': 126, 'name': 'animal'}, {'id': 127, 'name': 'bicycle'}, {'id': 129, 'name': 'dishwasher'}, {'id': 130, 'name': 'screen'}, {'id': 132, 'name': 'sculpture'}, {'id': 133, 'name': 'hood'}, {'id': 134, 'name': 'sconce'}, {'id': 135, 'name': 'vase'}, {'id': 136, 'name': 'traffic light'}, {'id': 137, 'name': 'tray'}, {'id': 138, 'name': 'ashcan'}, {'id': 139, 'name': 'fan'}, {'id': 142, 'name': 'plate'}, {'id': 143, 'name': 'monitor'}, {'id': 144, 'name': 'bulletin board'}, {'id': 146, 'name': 'radiator'}, {'id': 147, 'name': 'glass'}, {'id': 148, 'name': 'clock'}, {'id': 149, 'name': 'flag'}]


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "ade20k_instance_train": (
        "ADEChallengeData2016/images/training",
        "ADEChallengeData2016/ade20k_instance_train.json",
    ),
    "ade20k_instance_val": (
        "ADEChallengeData2016/images/validation",
        "ADEChallengeData2016/ade20k_instance_val.json",
    ),
}


def _get_ade_instances_meta():
    thing_ids = [k["id"] for k in ADE_CATEGORIES]
    assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ADE_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_ade20k_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ade_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_instance(_root)
