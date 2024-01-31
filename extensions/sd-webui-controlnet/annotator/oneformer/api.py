import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from annotator.oneformer.detectron2.config import get_cfg
from annotator.oneformer.detectron2.projects.deeplab import add_deeplab_config
from annotator.oneformer.detectron2.data import MetadataCatalog

from annotator.oneformer.oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)

from annotator.oneformer.oneformer.demo.defaults import DefaultPredictor
from annotator.oneformer.oneformer.demo.visualizer import Visualizer, ColorMode


def make_detectron2_model(config_path, ckpt_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.freeze()
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused")
    return DefaultPredictor(cfg), metadata


def semantic_run(img, predictor, metadata):
    predictions = predictor(img[:, :, ::-1], "semantic")  # Predictor of OneFormer must use BGR image !!!
    visualizer_map = Visualizer(img, is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).cpu(), alpha=1, is_text=False).get_image()
    return out_map
