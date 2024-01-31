# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_ade20k_panoptic.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import os

from annotator.oneformer.detectron2.data import DatasetCatalog, MetadataCatalog
from annotator.oneformer.detectron2.utils.file_io import PathManager

ADE20K_150_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "building"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "sky"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "floor"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "tree"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "ceiling"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "road, route"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "bed"},
    {"color": [230, 230, 230], "id": 8, "isthing": 1, "name": "window "},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "grass"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "cabinet"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "sidewalk, pavement"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "person"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "earth, ground"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "door"},
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "table"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "mountain, mount"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "plant"},
    {"color": [255, 51, 7], "id": 18, "isthing": 1, "name": "curtain"},
    {"color": [204, 70, 3], "id": 19, "isthing": 1, "name": "chair"},
    {"color": [0, 102, 200], "id": 20, "isthing": 1, "name": "car"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "water"},
    {"color": [255, 6, 51], "id": 22, "isthing": 1, "name": "painting, picture"},
    {"color": [11, 102, 255], "id": 23, "isthing": 1, "name": "sofa"},
    {"color": [255, 7, 71], "id": 24, "isthing": 1, "name": "shelf"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "house"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "sea"},
    {"color": [220, 220, 220], "id": 27, "isthing": 1, "name": "mirror"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "rug"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "field"},
    {"color": [8, 255, 214], "id": 30, "isthing": 1, "name": "armchair"},
    {"color": [7, 255, 224], "id": 31, "isthing": 1, "name": "seat"},
    {"color": [255, 184, 6], "id": 32, "isthing": 1, "name": "fence"},
    {"color": [10, 255, 71], "id": 33, "isthing": 1, "name": "desk"},
    {"color": [255, 41, 10], "id": 34, "isthing": 0, "name": "rock, stone"},
    {"color": [7, 255, 255], "id": 35, "isthing": 1, "name": "wardrobe, closet, press"},
    {"color": [224, 255, 8], "id": 36, "isthing": 1, "name": "lamp"},
    {"color": [102, 8, 255], "id": 37, "isthing": 1, "name": "tub"},
    {"color": [255, 61, 6], "id": 38, "isthing": 1, "name": "rail"},
    {"color": [255, 194, 7], "id": 39, "isthing": 1, "name": "cushion"},
    {"color": [255, 122, 8], "id": 40, "isthing": 0, "name": "base, pedestal, stand"},
    {"color": [0, 255, 20], "id": 41, "isthing": 1, "name": "box"},
    {"color": [255, 8, 41], "id": 42, "isthing": 1, "name": "column, pillar"},
    {"color": [255, 5, 153], "id": 43, "isthing": 1, "name": "signboard, sign"},
    {
        "color": [6, 51, 255],
        "id": 44,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [235, 12, 255], "id": 45, "isthing": 1, "name": "counter"},
    {"color": [160, 150, 20], "id": 46, "isthing": 0, "name": "sand"},
    {"color": [0, 163, 255], "id": 47, "isthing": 1, "name": "sink"},
    {"color": [140, 140, 140], "id": 48, "isthing": 0, "name": "skyscraper"},
    {"color": [250, 10, 15], "id": 49, "isthing": 1, "name": "fireplace"},
    {"color": [20, 255, 0], "id": 50, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [31, 255, 0], "id": 51, "isthing": 0, "name": "grandstand, covered stand"},
    {"color": [255, 31, 0], "id": 52, "isthing": 0, "name": "path"},
    {"color": [255, 224, 0], "id": 53, "isthing": 1, "name": "stairs"},
    {"color": [153, 255, 0], "id": 54, "isthing": 0, "name": "runway"},
    {"color": [0, 0, 255], "id": 55, "isthing": 1, "name": "case, display case, showcase, vitrine"},
    {
        "color": [255, 71, 0],
        "id": 56,
        "isthing": 1,
        "name": "pool table, billiard table, snooker table",
    },
    {"color": [0, 235, 255], "id": 57, "isthing": 1, "name": "pillow"},
    {"color": [0, 173, 255], "id": 58, "isthing": 1, "name": "screen door, screen"},
    {"color": [31, 0, 255], "id": 59, "isthing": 0, "name": "stairway, staircase"},
    {"color": [11, 200, 200], "id": 60, "isthing": 0, "name": "river"},
    {"color": [255, 82, 0], "id": 61, "isthing": 0, "name": "bridge, span"},
    {"color": [0, 255, 245], "id": 62, "isthing": 1, "name": "bookcase"},
    {"color": [0, 61, 255], "id": 63, "isthing": 0, "name": "blind, screen"},
    {"color": [0, 255, 112], "id": 64, "isthing": 1, "name": "coffee table"},
    {
        "color": [0, 255, 133],
        "id": 65,
        "isthing": 1,
        "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
    },
    {"color": [255, 0, 0], "id": 66, "isthing": 1, "name": "flower"},
    {"color": [255, 163, 0], "id": 67, "isthing": 1, "name": "book"},
    {"color": [255, 102, 0], "id": 68, "isthing": 0, "name": "hill"},
    {"color": [194, 255, 0], "id": 69, "isthing": 1, "name": "bench"},
    {"color": [0, 143, 255], "id": 70, "isthing": 1, "name": "countertop"},
    {"color": [51, 255, 0], "id": 71, "isthing": 1, "name": "stove"},
    {"color": [0, 82, 255], "id": 72, "isthing": 1, "name": "palm, palm tree"},
    {"color": [0, 255, 41], "id": 73, "isthing": 1, "name": "kitchen island"},
    {"color": [0, 255, 173], "id": 74, "isthing": 1, "name": "computer"},
    {"color": [10, 0, 255], "id": 75, "isthing": 1, "name": "swivel chair"},
    {"color": [173, 255, 0], "id": 76, "isthing": 1, "name": "boat"},
    {"color": [0, 255, 153], "id": 77, "isthing": 0, "name": "bar"},
    {"color": [255, 92, 0], "id": 78, "isthing": 1, "name": "arcade machine"},
    {"color": [255, 0, 255], "id": 79, "isthing": 0, "name": "hovel, hut, hutch, shack, shanty"},
    {"color": [255, 0, 245], "id": 80, "isthing": 1, "name": "bus"},
    {"color": [255, 0, 102], "id": 81, "isthing": 1, "name": "towel"},
    {"color": [255, 173, 0], "id": 82, "isthing": 1, "name": "light"},
    {"color": [255, 0, 20], "id": 83, "isthing": 1, "name": "truck"},
    {"color": [255, 184, 184], "id": 84, "isthing": 0, "name": "tower"},
    {"color": [0, 31, 255], "id": 85, "isthing": 1, "name": "chandelier"},
    {"color": [0, 255, 61], "id": 86, "isthing": 1, "name": "awning, sunshade, sunblind"},
    {"color": [0, 71, 255], "id": 87, "isthing": 1, "name": "street lamp"},
    {"color": [255, 0, 204], "id": 88, "isthing": 1, "name": "booth"},
    {"color": [0, 255, 194], "id": 89, "isthing": 1, "name": "tv"},
    {"color": [0, 255, 82], "id": 90, "isthing": 1, "name": "plane"},
    {"color": [0, 10, 255], "id": 91, "isthing": 0, "name": "dirt track"},
    {"color": [0, 112, 255], "id": 92, "isthing": 1, "name": "clothes"},
    {"color": [51, 0, 255], "id": 93, "isthing": 1, "name": "pole"},
    {"color": [0, 194, 255], "id": 94, "isthing": 0, "name": "land, ground, soil"},
    {
        "color": [0, 122, 255],
        "id": 95,
        "isthing": 1,
        "name": "bannister, banister, balustrade, balusters, handrail",
    },
    {
        "color": [0, 255, 163],
        "id": 96,
        "isthing": 0,
        "name": "escalator, moving staircase, moving stairway",
    },
    {
        "color": [255, 153, 0],
        "id": 97,
        "isthing": 1,
        "name": "ottoman, pouf, pouffe, puff, hassock",
    },
    {"color": [0, 255, 10], "id": 98, "isthing": 1, "name": "bottle"},
    {"color": [255, 112, 0], "id": 99, "isthing": 0, "name": "buffet, counter, sideboard"},
    {
        "color": [143, 255, 0],
        "id": 100,
        "isthing": 0,
        "name": "poster, posting, placard, notice, bill, card",
    },
    {"color": [82, 0, 255], "id": 101, "isthing": 0, "name": "stage"},
    {"color": [163, 255, 0], "id": 102, "isthing": 1, "name": "van"},
    {"color": [255, 235, 0], "id": 103, "isthing": 1, "name": "ship"},
    {"color": [8, 184, 170], "id": 104, "isthing": 1, "name": "fountain"},
    {
        "color": [133, 0, 255],
        "id": 105,
        "isthing": 0,
        "name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    },
    {"color": [0, 255, 92], "id": 106, "isthing": 0, "name": "canopy"},
    {
        "color": [184, 0, 255],
        "id": 107,
        "isthing": 1,
        "name": "washer, automatic washer, washing machine",
    },
    {"color": [255, 0, 31], "id": 108, "isthing": 1, "name": "plaything, toy"},
    {"color": [0, 184, 255], "id": 109, "isthing": 0, "name": "pool"},
    {"color": [0, 214, 255], "id": 110, "isthing": 1, "name": "stool"},
    {"color": [255, 0, 112], "id": 111, "isthing": 1, "name": "barrel, cask"},
    {"color": [92, 255, 0], "id": 112, "isthing": 1, "name": "basket, handbasket"},
    {"color": [0, 224, 255], "id": 113, "isthing": 0, "name": "falls"},
    {"color": [112, 224, 255], "id": 114, "isthing": 0, "name": "tent"},
    {"color": [70, 184, 160], "id": 115, "isthing": 1, "name": "bag"},
    {"color": [163, 0, 255], "id": 116, "isthing": 1, "name": "minibike, motorbike"},
    {"color": [153, 0, 255], "id": 117, "isthing": 0, "name": "cradle"},
    {"color": [71, 255, 0], "id": 118, "isthing": 1, "name": "oven"},
    {"color": [255, 0, 163], "id": 119, "isthing": 1, "name": "ball"},
    {"color": [255, 204, 0], "id": 120, "isthing": 1, "name": "food, solid food"},
    {"color": [255, 0, 143], "id": 121, "isthing": 1, "name": "step, stair"},
    {"color": [0, 255, 235], "id": 122, "isthing": 0, "name": "tank, storage tank"},
    {"color": [133, 255, 0], "id": 123, "isthing": 1, "name": "trade name"},
    {"color": [255, 0, 235], "id": 124, "isthing": 1, "name": "microwave"},
    {"color": [245, 0, 255], "id": 125, "isthing": 1, "name": "pot"},
    {"color": [255, 0, 122], "id": 126, "isthing": 1, "name": "animal"},
    {"color": [255, 245, 0], "id": 127, "isthing": 1, "name": "bicycle"},
    {"color": [10, 190, 212], "id": 128, "isthing": 0, "name": "lake"},
    {"color": [214, 255, 0], "id": 129, "isthing": 1, "name": "dishwasher"},
    {"color": [0, 204, 255], "id": 130, "isthing": 1, "name": "screen"},
    {"color": [20, 0, 255], "id": 131, "isthing": 0, "name": "blanket, cover"},
    {"color": [255, 255, 0], "id": 132, "isthing": 1, "name": "sculpture"},
    {"color": [0, 153, 255], "id": 133, "isthing": 1, "name": "hood, exhaust hood"},
    {"color": [0, 41, 255], "id": 134, "isthing": 1, "name": "sconce"},
    {"color": [0, 255, 204], "id": 135, "isthing": 1, "name": "vase"},
    {"color": [41, 0, 255], "id": 136, "isthing": 1, "name": "traffic light"},
    {"color": [41, 255, 0], "id": 137, "isthing": 1, "name": "tray"},
    {"color": [173, 0, 255], "id": 138, "isthing": 1, "name": "trash can"},
    {"color": [0, 245, 255], "id": 139, "isthing": 1, "name": "fan"},
    {"color": [71, 0, 255], "id": 140, "isthing": 0, "name": "pier"},
    {"color": [122, 0, 255], "id": 141, "isthing": 0, "name": "crt screen"},
    {"color": [0, 255, 184], "id": 142, "isthing": 1, "name": "plate"},
    {"color": [0, 92, 255], "id": 143, "isthing": 1, "name": "monitor"},
    {"color": [184, 255, 0], "id": 144, "isthing": 1, "name": "bulletin board"},
    {"color": [0, 133, 255], "id": 145, "isthing": 0, "name": "shower"},
    {"color": [255, 214, 0], "id": 146, "isthing": 1, "name": "radiator"},
    {"color": [25, 194, 194], "id": 147, "isthing": 1, "name": "glass, drinking glass"},
    {"color": [102, 255, 0], "id": 148, "isthing": 1, "name": "clock"},
    {"color": [92, 0, 255], "id": 149, "isthing": 1, "name": "flag"},
]

ADE20k_COLORS = [k["color"] for k in ADE20K_150_CATEGORIES]

MetadataCatalog.get("ade20k_sem_seg_train").set(
    stuff_colors=ADE20k_COLORS[:],
)

MetadataCatalog.get("ade20k_sem_seg_val").set(
    stuff_colors=ADE20k_COLORS[:],
)


def load_ade20k_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
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
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_ade20k_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
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
        lambda: load_ade20k_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="ade20k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "ade20k_panoptic_train": (
        "ADEChallengeData2016/images/training",
        "ADEChallengeData2016/ade20k_panoptic_train",
        "ADEChallengeData2016/ade20k_panoptic_train.json",
        "ADEChallengeData2016/annotations_detectron2/training",
        "ADEChallengeData2016/ade20k_instance_train.json",
    ),
    "ade20k_panoptic_val": (
        "ADEChallengeData2016/images/validation",
        "ADEChallengeData2016/ade20k_panoptic_val",
        "ADEChallengeData2016/ade20k_panoptic_val.json",
        "ADEChallengeData2016/annotations_detectron2/validation",
        "ADEChallengeData2016/ade20k_instance_val.json",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in ADE20K_150_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in ADE20K_150_CATEGORIES]
    stuff_colors = [k["color"] for k in ADE20K_150_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(ADE20K_150_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_ade20k_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root, instance_json),
    ) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_ade20k_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, instance_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_panoptic(_root)
