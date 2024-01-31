# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
from annotator.oneformer.detectron2.data.datasets.builtin_meta import  _get_builtin_metadata
from annotator.oneformer.detectron2.data.datasets.coco import register_coco_instances


_PREDEFINED_SPLITS_COCO = {
    "coco_2017_val_panoptic2instance": ("coco/val2017", "coco/annotations/panoptic2instances_val2017.json"),
}


def register_panoptic2instances_coco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_panoptic2instances_coco(_root)