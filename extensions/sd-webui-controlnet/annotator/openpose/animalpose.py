import cv2
import numpy as np
from typing import List

from .cv_ox_det import inference_detector
from .cv_ox_pose import inference_pose

from .types import AnimalPoseResult, Keypoint


def draw_animalposes(animals: List[List[Keypoint]], H: int, W: int) -> np.ndarray:
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for animal_pose in animals:
        canvas = draw_animalpose(canvas, animal_pose)
    return canvas


def draw_animalpose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    # order of the keypoints for AP10k and a standardized list of colors for limbs
    keypointPairsList = [
        (1, 2),
        (2, 3),
        (1, 3),
        (3, 4),
        (4, 9),
        (9, 10),
        (10, 11),
        (4, 6),
        (6, 7),
        (7, 8),
        (4, 5),
        (5, 15),
        (15, 16),
        (16, 17),
        (5, 12),
        (12, 13),
        (13, 14),
    ]
    colorsList = [
        (255, 255, 255),
        (100, 255, 100),
        (150, 255, 255),
        (100, 50, 255),
        (50, 150, 200),
        (0, 255, 255),
        (0, 150, 0),
        (0, 0, 255),
        (0, 0, 150),
        (255, 50, 255),
        (255, 0, 255),
        (255, 0, 0),
        (150, 0, 0),
        (255, 255, 100),
        (0, 150, 0),
        (255, 255, 0),
        (150, 150, 150),
    ]  # 16 colors needed

    for ind, (i, j) in enumerate(keypointPairsList):
        p1 = keypoints[i - 1]
        p2 = keypoints[j - 1]

        if p1 is not None and p2 is not None:
            cv2.line(
                canvas,
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                colorsList[ind],
                5,
            )
    return canvas


class AnimalPose:
    def __init__(
        self,
        onnx_det: str,
        onnx_pose: str,
    ):
        self.onnx_det = onnx_det
        self.onnx_pose = onnx_pose
        self.model_input_size = (256, 256)

        # Always loads to CPU to avoid building OpenCV.
        device = 'cpu'
        backend = cv2.dnn.DNN_BACKEND_OPENCV if device == 'cpu' else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        providers = cv2.dnn.DNN_TARGET_CPU if device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA

        self.session_det = cv2.dnn.readNetFromONNX(onnx_det)
        self.session_det.setPreferableBackend(backend)
        self.session_det.setPreferableTarget(providers)

        self.session_pose = cv2.dnn.readNetFromONNX(onnx_pose)
        self.session_pose.setPreferableBackend(backend)
        self.session_pose.setPreferableTarget(providers)

    def __call__(self, oriImg) -> List[AnimalPoseResult]:
        detect_classes = list(
            range(14, 23 + 1)
        )  # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

        det_result = inference_detector(
            self.session_det,
            oriImg,
            detect_classes=detect_classes,
        )

        if (det_result is None) or (det_result.shape[0] == 0):
            return []

        keypoint_sets, scores = inference_pose(
            self.session_pose,
            det_result,
            oriImg,
            self.model_input_size,
        )

        animals = []
        for idx, keypoints in enumerate(keypoint_sets):
            score = scores[idx, ..., None]
            score[score > 1.0] = 1.0
            score[score < 0.0] = 0.0
            animals.append(
                [
                    Keypoint(x, y, c)
                    for x, y, c in np.concatenate((keypoints, score), axis=-1).tolist()
                ]
            )

        return animals
