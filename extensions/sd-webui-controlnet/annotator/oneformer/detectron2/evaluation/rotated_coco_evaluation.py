# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import numpy as np
import os
import torch
from annotator.oneformer.pycocotools.cocoeval import COCOeval, maskUtils

from annotator.oneformer.detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
from annotator.oneformer.detectron2.utils.file_io import PathManager

from .coco_evaluation import COCOEvaluator


class RotatedCOCOeval(COCOeval):
    @staticmethod
    def is_rotated(box_list):
        if type(box_list) == np.ndarray:
            return box_list.shape[1] == 5
        elif type(box_list) == list:
            if box_list == []:  # cannot decide the box_dim
                return False
            return np.all(
                np.array(
                    [
                        (len(obj) == 5) and ((type(obj) == list) or (type(obj) == np.ndarray))
                        for obj in box_list
                    ]
                )
            )
        return False

    @staticmethod
    def boxlist_to_tensor(boxlist, output_box_dim):
        if type(boxlist) == np.ndarray:
            box_tensor = torch.from_numpy(boxlist)
        elif type(boxlist) == list:
            if boxlist == []:
                return torch.zeros((0, output_box_dim), dtype=torch.float32)
            else:
                box_tensor = torch.FloatTensor(boxlist)
        else:
            raise Exception("Unrecognized boxlist type")

        input_box_dim = box_tensor.shape[1]
        if input_box_dim != output_box_dim:
            if input_box_dim == 4 and output_box_dim == 5:
                box_tensor = BoxMode.convert(box_tensor, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS)
            else:
                raise Exception(
                    "Unable to convert from {}-dim box to {}-dim box".format(
                        input_box_dim, output_box_dim
                    )
                )
        return box_tensor

    def compute_iou_dt_gt(self, dt, gt, is_crowd):
        if self.is_rotated(dt) or self.is_rotated(gt):
            # TODO: take is_crowd into consideration
            assert all(c == 0 for c in is_crowd)
            dt = RotatedBoxes(self.boxlist_to_tensor(dt, output_box_dim=5))
            gt = RotatedBoxes(self.boxlist_to_tensor(gt, output_box_dim=5))
            return pairwise_iou_rotated(dt, gt)
        else:
            # This is the same as the classical COCO evaluation
            return maskUtils.iou(dt, gt, is_crowd)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        assert p.iouType == "bbox", "unsupported iouType for iou computation"

        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Note: this function is copied from cocoeval.py in cocoapi
        # and the major difference is here.
        ious = self.compute_iou_dt_gt(d, g, iscrowd)
        return ious


class RotatedCOCOEvaluator(COCOEvaluator):
    """
    Evaluate object proposal/instance detection outputs using COCO-like metrics and APIs,
    with rotated boxes support.
    Note: this uses IOU only and does not consider angle differences.
    """

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)

                prediction["instances"] = self.instances_to_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def instances_to_json(self, instances, img_id):
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        if boxes.shape[1] == 4:
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            results.append(result)
        return results

    def _eval_predictions(self, predictions, img_ids=None):  # img_ids: unused
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")

        assert self._tasks is None or set(self._tasks) == {
            "bbox"
        }, "[RotatedCOCOEvaluator] Only bbox evaluation is supported"
        coco_eval = (
            self._evaluate_predictions_on_coco(self._coco_api, coco_results)
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        task = "bbox"
        res = self._derive_coco_results(
            coco_eval, task, class_names=self._metadata.get("thing_classes")
        )
        self._results[task] = res

    def _evaluate_predictions_on_coco(self, coco_gt, coco_results):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        coco_dt = coco_gt.loadRes(coco_results)

        # Only bbox is supported for now
        coco_eval = RotatedCOCOeval(coco_gt, coco_dt, iouType="bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval
