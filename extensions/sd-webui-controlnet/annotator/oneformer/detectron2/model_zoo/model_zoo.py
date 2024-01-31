# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Optional
import pkg_resources
import torch

from annotator.oneformer.detectron2.checkpoint import DetectionCheckpointer
from annotator.oneformer.detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from annotator.oneformer.detectron2.modeling import build_model


class _ModelZooUrls(object):
    """
    Mapping from names to officially released Detectron2 pre-trained models.
    """

    S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    # format: {config_path.yaml} -> model_id/model_final_{commit}.pkl
    CONFIG_PATH_TO_URL_SUFFIX = {
        # COCO Detection with Faster R-CNN
        "COCO-Detection/faster_rcnn_R_50_C4_1x": "137257644/model_final_721ade.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_1x": "137847829/model_final_51d356.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_1x": "137257794/model_final_b275ba.pkl",
        "COCO-Detection/faster_rcnn_R_50_C4_3x": "137849393/model_final_f97cb7.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_3x": "137849425/model_final_68d202.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x": "137849458/model_final_280758.pkl",
        "COCO-Detection/faster_rcnn_R_101_C4_3x": "138204752/model_final_298dad.pkl",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x": "138204841/model_final_3e0943.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x": "137851257/model_final_f6e8b1.pkl",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x": "139173657/model_final_68b088.pkl",
        # COCO Detection with RetinaNet
        "COCO-Detection/retinanet_R_50_FPN_1x": "190397773/model_final_bfca0b.pkl",
        "COCO-Detection/retinanet_R_50_FPN_3x": "190397829/model_final_5bd44e.pkl",
        "COCO-Detection/retinanet_R_101_FPN_3x": "190397697/model_final_971ab9.pkl",
        # COCO Detection with RPN and Fast R-CNN
        "COCO-Detection/rpn_R_50_C4_1x": "137258005/model_final_450694.pkl",
        "COCO-Detection/rpn_R_50_FPN_1x": "137258492/model_final_02ce48.pkl",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x": "137635226/model_final_e5f7ce.pkl",
        # COCO Instance Segmentation Baselines with Mask R-CNN
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x": "137259246/model_final_9243eb.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x": "137260150/model_final_4f86c3.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x": "137260431/model_final_a54504.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x": "137849525/model_final_4ce675.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x": "137849551/model_final_84107b.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x": "137849600/model_final_f10217.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x": "138363239/model_final_a2914c.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x": "138363294/model_final_0464b7.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x": "138205316/model_final_a3ec72.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x": "139653917/model_final_2d9806.pkl",  # noqa
        # New baselines using Large-Scale Jitter and Longer Training Schedule
        "new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ": "42047764/model_final_bb69de.pkl",
        "new_baselines/mask_rcnn_R_50_FPN_200ep_LSJ": "42047638/model_final_89a8d3.pkl",
        "new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ": "42019571/model_final_14d201.pkl",
        "new_baselines/mask_rcnn_R_101_FPN_100ep_LSJ": "42025812/model_final_4f7b58.pkl",
        "new_baselines/mask_rcnn_R_101_FPN_200ep_LSJ": "42131867/model_final_0bb7ae.pkl",
        "new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ": "42073830/model_final_f96b26.pkl",
        "new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_100ep_LSJ": "42047771/model_final_b7fbab.pkl",  # noqa
        "new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_200ep_LSJ": "42132721/model_final_5d87c1.pkl",  # noqa
        "new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ": "42025447/model_final_f1362d.pkl",  # noqa
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_100ep_LSJ": "42047784/model_final_6ba57e.pkl",  # noqa
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_200ep_LSJ": "42047642/model_final_27b9c1.pkl",  # noqa
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ": "42045954/model_final_ef3a80.pkl",  # noqa
        # COCO Person Keypoint Detection Baselines with Keypoint R-CNN
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x": "137261548/model_final_04e291.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x": "137849621/model_final_a6e10b.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x": "138363331/model_final_997cc7.pkl",
        "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x": "139686956/model_final_5ad38f.pkl",
        # COCO Panoptic Segmentation Baselines with Panoptic FPN
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x": "139514544/model_final_dbfeb4.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x": "139514569/model_final_c10459.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x": "139514519/model_final_cafdb1.pkl",
        # LVIS Instance Segmentation Baselines with Mask R-CNN
        "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x": "144219072/model_final_571f7c.pkl",  # noqa
        "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x": "144219035/model_final_824ab5.pkl",  # noqa
        "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x": "144219108/model_final_5e3439.pkl",  # noqa
        # Cityscapes & Pascal VOC Baselines
        "Cityscapes/mask_rcnn_R_50_FPN": "142423278/model_final_af9cf5.pkl",
        "PascalVOC-Detection/faster_rcnn_R_50_C4": "142202221/model_final_b1acc2.pkl",
        # Other Settings
        "Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5": "138602867/model_final_65c703.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5": "144998336/model_final_821d0b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_1x": "138602847/model_final_e9d89b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_3x": "144998488/model_final_480dd8.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_syncbn": "169527823/model_final_3b3c51.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_gn": "138602888/model_final_dc5d9e.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn": "138602908/model_final_01ca85.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_9x_gn": "183808979/model_final_da7b4c.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn": "184226666/model_final_5ce33e.pkl",
        "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x": "139797668/model_final_be35db.pkl",
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv": "18131413/model_0039999_e76410.pkl",  # noqa
        # D1 Comparisons
        "Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x": "137781054/model_final_7ab50c.pkl",  # noqa
        "Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x": "137781281/model_final_62ca52.pkl",  # noqa
        "Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x": "137781195/model_final_cce136.pkl",
    }

    @staticmethod
    def query(config_path: str) -> Optional[str]:
        """
        Args:
            config_path: relative config filename
        """
        name = config_path.replace(".yaml", "").replace(".py", "")
        if name in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
            suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[name]
            return _ModelZooUrls.S3_PREFIX + name + "/" + suffix
        return None


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: a URL to the model
    """
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError("Pretrained model for {} is not available!".format(config_path))
    return url


def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_config(config_path, trained: bool = False):
    """
    Returns a config object for a model in model zoo.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): If True, will set ``MODEL.WEIGHTS`` to trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.

    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """
    cfg_file = get_config_file(config_path)
    if cfg_file.endswith(".yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        if trained:
            cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
        return cfg
    elif cfg_file.endswith(".py"):
        cfg = LazyConfig.load(cfg_file)
        if trained:
            url = get_checkpoint_url(config_path)
            if "train" in cfg and "init_checkpoint" in cfg.train:
                cfg.train.init_checkpoint = url
            else:
                raise NotImplementedError
        return cfg


def get(config_path, trained: bool = False, device: Optional[str] = None):
    """
    Get a model specified by relative path under Detectron2's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to detectron2's "configs/"
            directory, e.g., "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.

    Returns:
        nn.Module: a detectron2 model. Will be in training mode.

    Example:
    ::
        from annotator.oneformer.detectron2 import model_zoo
        model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
    """
    cfg = get_config(config_path, trained)
    if device is None and not torch.cuda.is_available():
        device = "cpu"
    if device is not None and isinstance(cfg, CfgNode):
        cfg.MODEL.DEVICE = device

    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        model = instantiate(cfg.model)
        if device is not None:
            model = model.to(device)
        if "train" in cfg and "init_checkpoint" in cfg.train:
            DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    return model
