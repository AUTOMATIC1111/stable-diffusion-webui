# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from annotator.oneformer.detectron2.config import CfgNode
from annotator.oneformer.detectron2.solver import LRScheduler
from annotator.oneformer.detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler

from .lr_scheduler import WarmupPolyLR


def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        )
    else:
        return build_d2_lr_scheduler(cfg, optimizer)
