# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json
import os

from annotator.oneformer.detectron2.data import DatasetCatalog, MetadataCatalog
from annotator.oneformer.detectron2.utils.file_io import PathManager

from .coco import load_coco_json, load_sem_seg

__all__ = ["register_coco_panoptic", "register_coco_panoptic_separated"]


def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_coco_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_coco_panoptic_separated(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    It follows the setting used by the PanopticFPN paper:

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name + "_separated"
    DatasetCatalog.register(
        panoptic_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, panoptic_name),
            load_sem_seg(sem_seg_root, image_root),
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        sem_seg_root=sem_seg_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        **metadata,
    )

    semantic_name = name + "_stuffonly"
    DatasetCatalog.register(semantic_name, lambda: load_sem_seg(sem_seg_root, image_root))
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        image_root=image_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        **metadata,
    )


def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.

    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.

    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results


if __name__ == "__main__":
    """
    Test the COCO panoptic dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco_panoptic \
            path/to/image_root path/to/panoptic_root path/to/panoptic_json dataset_name 10

        "dataset_name" can be "coco_2017_train_panoptic", or other
        pre-registered ones
    """
    from annotator.oneformer.detectron2.utils.logger import setup_logger
    from annotator.oneformer.detectron2.utils.visualizer import Visualizer
    import annotator.oneformer.detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image
    import numpy as np

    logger = setup_logger(name=__name__)
    assert sys.argv[4] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[4])

    dicts = load_coco_panoptic_json(sys.argv[3], sys.argv[1], sys.argv[2], meta.as_dict())
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    num_imgs_to_vis = int(sys.argv[5])
    for i, d in enumerate(dicts):
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
        if i + 1 >= num_imgs_to_vis:
            break
