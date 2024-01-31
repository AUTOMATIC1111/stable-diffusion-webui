import numpy as np
import cv2
import torch

import os
from modules import devices
from annotator.annotator_path import models_path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_top_down_pose_model
from mmpose.apis import init_pose_model, process_mmdet_results, vis_pose_result


def preprocessing(image, device):
    # Resize
    scale = 640 / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(104.008),
            float(116.669),
            float(122.675),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.1,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1):
    """Draw keypoints and links on an image.
    Args:
            img (ndarry): The image to draw poses on.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    for idx, kpts in enumerate(pose_result):
        if idx > 1:
            continue
        kpts = kpts['keypoints']
        # print(kpts)
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                cv2.circle(img, (int(x_coord), int(y_coord)),
                           radius, color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0 or pos1[1] >= img_h or pos2[0] <= 0
                        or pos2[0] >= img_w or pos2[1] <= 0 or pos2[1] >= img_h or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


human_det, pose_model = None, None
det_model_path = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_model_path = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

modeldir = os.path.join(models_path, "keypose")
old_modeldir = os.path.dirname(os.path.realpath(__file__))

det_config = 'faster_rcnn_r50_fpn_coco.py'
pose_config = 'hrnet_w48_coco_256x192.py'

det_checkpoint = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_checkpoint = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_cat_id = 1
bbox_thr = 0.2

skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]

pose_kpt_color = [
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
    [0, 255, 0],
    [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]
]

pose_link_color = [
    [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0],
    [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255], [51, 153, 255], [51, 153, 255]
]

def find_download_model(checkpoint, remote_path):
    modelpath = os.path.join(modeldir, checkpoint)
    old_modelpath = os.path.join(old_modeldir, checkpoint)
        
    if os.path.exists(old_modelpath):
        modelpath = old_modelpath
    elif not os.path.exists(modelpath):
        from basicsr.utils.download_util import load_file_from_url
        load_file_from_url(remote_path, model_dir=modeldir)
        
    return modelpath

def apply_keypose(input_image):
    global human_det, pose_model
    if netNetwork is None:
        det_model_local = find_download_model(det_checkpoint, det_model_path)
        hrnet_model_local = find_download_model(pose_checkpoint, pose_model_path)
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        human_det = init_detector(det_config_mmcv, det_model_local, device=devices.get_device_for("controlnet"))
        pose_model = init_pose_model(pose_config_mmcv, hrnet_model_local, device=devices.get_device_for("controlnet"))

    assert input_image.ndim == 3
    input_image = input_image.copy()
    with torch.no_grad():
        image = torch.from_numpy(input_image).float().to(devices.get_device_for("controlnet"))
        image = image / 255.0
        mmdet_results = inference_detector(human_det, image)
        
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)
        
        return_heatmap = False
        dataset = pose_model.cfg.data['test']['type']
        
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=None,
            return_heatmap=return_heatmap,
            outputs=output_layer_names
        )
        
        im_keypose_out = imshow_keypoints(
            image,
            pose_results,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            radius=2,
            thickness=2
        )
        im_keypose_out = im_keypose_out.astype(np.uint8)

        # image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
        # edge = netNetwork(image_hed)[0]
        # edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return im_keypose_out


def unload_hed_model():
    global netNetwork
    if netNetwork is not None:
        netNetwork.cpu()
