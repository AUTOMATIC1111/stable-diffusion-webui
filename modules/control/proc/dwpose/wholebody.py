# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
from modules.shared import log

mmok = True

try:
    import mmcv # pylint: disable=unused-import
except ImportError as e:
    mmok = False
    log.error(f"Control processor DWPose: {e}")
try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    from mmpose.structures import merge_data_samples
except ImportError as e:
    mmok = False
    log.error(f"Control processor DWPose: {e}")

try:
    from mmdet.apis import inference_detector, init_detector
except ImportError as e:
    mmok = False
    log.error(f"Control processor DWPose: {e}")

    def inference_detector(*args, **kwargs):
        return lambda *args, **kwargs: None

if not mmok:
    log.error('Control processor DWPose: OpenMMLab is not installed')


class Wholebody:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu"):
        if not mmok:
            self.detector = lambda *args, **kwargs: None
            return None
        prefix = os.path.dirname(__file__)
        if det_config is None:
            det_config = "config/yolox_l_8xb8-300e_coco.py"
        if pose_config is None:
            pose_config = "config/dwpose-l_384x288.py"
        if not det_config.startswith('prefix'):
            det_config = os.path.join(prefix, det_config)
        if not pose_config.startswith('prefix'):
            pose_config = os.path.join(prefix, pose_config)
        if det_ckpt is None:
            det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        if pose_ckpt is None:
            pose_ckpt = "https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"
        # build detector
        self.detector = init_detector(det_config, det_ckpt, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_ckpt,
            device=device)

    def to(self, device):
        self.detector.to(device)
        self.pose_estimator.to(device)
        return self

    def __call__(self, oriImg):
        if not mmok:
            return None, None
        # predict bbox
        det_result = inference_detector(self.detector, oriImg)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)]
            # set NMS threshold
        bboxes = bboxes[nms(bboxes, 0.7), :4]
        # predict keypoints
        if len(bboxes) == 0:
            pose_results = inference_topdown(self.pose_estimator, oriImg)
        else:
            pose_results = inference_topdown(self.pose_estimator, oriImg, bboxes)
        preds = merge_data_samples(pose_results)
        preds = preds.pred_instances
        # preds = pose_results[0].pred_instances
        keypoints = preds.get('transformed_keypoints', preds.keypoints)
        if 'keypoint_scores' in preds:
            scores = preds.keypoint_scores
        else:
            scores = np.ones(keypoints.shape[:-1])
        if 'keypoints_visible' in preds:
            visible = preds.keypoints_visible
        else:
            visible = np.ones(keypoints.shape[:-1])
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None], visible[..., None]),
            axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info
        keypoints, scores, visible = keypoints_info[..., :2], keypoints_info[..., 2], keypoints_info[..., 3]
        return keypoints, scores
