# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import copy
import logging

import numpy as np
import torch

from annotator.oneformer.detectron2.data import MetadataCatalog
from annotator.oneformer.detectron2.config import configurable
from annotator.oneformer.detectron2.data import detection_utils as utils
from annotator.oneformer.detectron2.data import transforms as T
from annotator.oneformer.detectron2.structures import BitMasks, Instances
from annotator.oneformer.oneformer.utils.box_ops import masks_to_boxes
from annotator.oneformer.oneformer.data.tokenizer import SimpleTokenizer, Tokenize

__all__ = ["COCOUnifiedNewBaselineDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOUnifiedNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by OneFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        num_queries,
        tfm_gens,
        meta,
        image_format,
        max_seq_len,
        task_seq_len,
        semantic_prob,
        instance_prob,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOUnifiedNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.meta = meta
        self.ignore_label = self.meta.ignore_label
        self.num_queries = num_queries

        self.things = []
        for k,v in self.meta.thing_dataset_id_to_contiguous_id.items():
            self.things.append(v)
        self.class_names = self.meta.stuff_classes
        self.text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)
        self.semantic_prob = semantic_prob
        self.instance_prob = instance_prob

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])

        ret = {
            "is_train": is_train,
            "meta": meta,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "semantic_prob": cfg.INPUT.TASK_PROB.SEMANTIC,
            "instance_prob": cfg.INPUT.TASK_PROB.INSTANCE,
        }
        return ret
    
    def _get_semantic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)
        
        classes = []
        texts = ["a semantic photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask == False):
                    if class_id not in classes:
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                    else:
                        idx = classes.index(class_id)
                        masks[idx] += mask
                        masks[idx] = np.clip(masks[idx], 0, 1).astype(np.bool)
                    label[mask] = class_id

        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1
                    
        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            # Placeholder bounding boxes for stuff regions. Note that these are not used during training.
            instances.gt_bboxes = torch.stack([torch.tensor([0., 0., 1., 1.])] * instances.gt_masks.shape[0])
        return instances, texts, label
    
    def _get_instance_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)

        classes = []
        texts = ["an instance photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if class_id in self.things:
                if not segment_info["iscrowd"]:
                    mask = pan_seg_gt == segment_info["id"]
                    if not np.all(mask == False):
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                        label[mask] = class_id
        
        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
        return instances, texts, label
    
    def _get_panoptic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)

        classes = []
        texts = ["a panoptic photo"] * self.num_queries
        masks = []
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask == False):
                    cls_name = self.class_names[class_id]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                    label[mask] = class_id
        
        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
            for i in range(instances.gt_classes.shape[0]):
                # Placeholder bounding boxes for stuff regions. Note that these are not used during training.
                if instances.gt_classes[i].item() not in self.things:
                    instances.gt_bboxes[i] = torch.tensor([0., 0., 1., 1.])
        return instances, texts, label

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        else:
            sem_seg_gt = None
        
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id
            pan_seg_gt = rgb2id(pan_seg_gt)

        prob_task = np.random.uniform(0,1.)

        num_class_obj = {}

        for name in self.class_names:
            num_class_obj[name] = 0

        if prob_task < self.semantic_prob:
            task = "The task is semantic"
            instances, text, sem_seg = self._get_semantic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        elif prob_task < self.instance_prob:
            task = "The task is instance"
            instances, text, sem_seg = self._get_instance_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        else:
            task = "The task is panoptic"
            instances, text, sem_seg = self._get_panoptic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)


        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg).long()
        dataset_dict["instances"] = instances
        dataset_dict["orig_shape"] = image_shape
        dataset_dict["task"] = task
        dataset_dict["text"] = text
        dataset_dict["thing_ids"] = self.things

        return dataset_dict
