# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import warnings
from typing import Callable, List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from modules.control.util import HWC3, resize_image
from . import util
from .body import Body, BodyResult, Keypoint
from .face import Face
from .hand import Hand

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, body_estimation, hand_estimation=None, face_estimation=None):
        self.body_estimation = body_estimation
        self.hand_estimation = hand_estimation
        self.face_estimation = face_estimation

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, hand_filename=None, face_filename=None, cache_dir=None):

        if pretrained_model_or_path == "lllyasviel/ControlNet":
            filename = filename or "annotator/ckpts/body_pose_model.pth"
            hand_filename = hand_filename or "annotator/ckpts/hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = "lllyasviel/Annotators"
        else:
            filename = filename or "body_pose_model.pth"
            hand_filename = hand_filename or "hand_pose_model.pth"
            face_filename = face_filename or "facenet.pth"

            face_pretrained_model_or_path = pretrained_model_or_path

        if os.path.isdir(pretrained_model_or_path):
            body_model_path = os.path.join(pretrained_model_or_path, filename)
            hand_model_path = os.path.join(pretrained_model_or_path, hand_filename)
            face_model_path = os.path.join(face_pretrained_model_or_path, face_filename)
        else:
            body_model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
            hand_model_path = hf_hub_download(pretrained_model_or_path, hand_filename, cache_dir=cache_dir)
            face_model_path = hf_hub_download(face_pretrained_model_or_path, face_filename, cache_dir=cache_dir)

        body_estimation = Body(body_model_path)
        hand_estimation = Hand(hand_model_path)
        face_estimation = Face(face_model_path)

        return cls(body_estimation, hand_estimation, face_estimation)

    def to(self, device):
        self.body_estimation.to(device)
        self.hand_estimation.to(device)
        self.face_estimation.to(device)
        return self

    def detect_hands(self, body: BodyResult, oriImg) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)

                hand_result = [
                    Keypoint(x=peak[0], y=peak[1])
                    for peak in peaks
                ]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None

        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]

        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        candidate, subset = self.body_estimation(oriImg)
        bodies = self.body_estimation.format_body_result(candidate, subset)

        results = []
        for body in bodies:
            left_hand, right_hand, face = (None,) * 3
            if include_hand:
                left_hand, right_hand = self.detect_hands(body, oriImg)
            if include_face:
                face = self.detect_face(body, oriImg)

            results.append(PoseResult(BodyResult(
                keypoints=[
                    Keypoint(
                        x=keypoint.x / float(W),
                        y=keypoint.y / float(H)
                    ) if keypoint is not None else None
                    for keypoint in body.keypoints
                ],
                total_score=body.total_score,
                total_parts=body.total_parts
            ), left_hand, right_hand, face))

        return results

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape

        poses = self.detect_poses(input_image, include_hand, include_face)
        canvas = draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face)

        detected_map = canvas
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
