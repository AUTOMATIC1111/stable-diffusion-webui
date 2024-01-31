from typing import NamedTuple, List, Optional, Union


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


class BodyResult(NamedTuple):
    # Note: Using `Optional` instead of `|` operator as the ladder is a Python
    # 3.10 feature.
    # Annotator code should be Python 3.8 Compatible, as controlnet repo uses
    # Python 3.8 environment.
    # https://github.com/lllyasviel/ControlNet/blob/d3284fcd0972c510635a4f5abe2eeb71dc0de524/environment.yaml#L6
    keypoints: List[Optional[Keypoint]]
    total_score: float = 0.0
    total_parts: int = 0


HandResult = List[Keypoint]
FaceResult = List[Keypoint]
AnimalPoseResult = List[Keypoint]


class HumanPoseResult(NamedTuple):
    body: BodyResult
    left_hand: Optional[HandResult]
    right_hand: Optional[HandResult]
    face: Optional[FaceResult]
