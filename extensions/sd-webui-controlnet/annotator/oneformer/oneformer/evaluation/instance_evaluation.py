# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/evaluation/instance_evaluation.py
# ------------------------------------------------------------------------------

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import annotator.oneformer.pycocotools.mask as mask_util
import torch
from annotator.oneformer.pycocotools.coco import COCO
from annotator.oneformer.pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import annotator.oneformer.detectron2.utils.comm as comm
from annotator.oneformer.detectron2.config import CfgNode
from annotator.oneformer.detectron2.data import MetadataCatalog
from annotator.oneformer.detectron2.data.datasets.coco import convert_to_coco_json
from annotator.oneformer.detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from annotator.oneformer.detectron2.evaluation.fast_eval_api import COCOeval_opt
from annotator.oneformer.detectron2.structures import Boxes, BoxMode, pairwise_iou
from annotator.oneformer.detectron2.utils.file_io import PathManager
from annotator.oneformer.detectron2.utils.logger import create_small_table


# modified from COCOEvaluator for instance segmetnat
class InstanceSegEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            # all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            # num_classes = len(all_contiguous_ids)
            # assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                # assert category_id < num_classes, (
                #     f"A prediction has class={category_id}, "
                #     f"but the dataset only has {num_classes} classes and "
                #     f"predicted class id should be in [0, {num_classes - 1}]."
                # )
                assert category_id in reverse_id_mapping, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has class ids in {dataset_id_to_contiguous_id}."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res
