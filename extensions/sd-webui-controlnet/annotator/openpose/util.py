import math
import numpy as np
import matplotlib
import cv2
from typing import List, Tuple, Union, Optional

from .body import BodyResult, Keypoint

eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


def is_normalized(keypoints: List[Optional[Keypoint]]) -> bool:
    point_normalized = [
        0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1 
        for k in keypoints 
        if k is not None
    ]
    if not point_normalized:
        return False
    return all(point_normalized)

    
def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not keypoints:
        return canvas
    
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    for keypoint in keypoints:
        if keypoint is None:
            continue
        
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(body: BodyResult, oriImg) -> List[Tuple[int, int, int, bool]]:
    """
    Detect hands in the input body pose keypoints and calculate the bounding box for each hand.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        List[Tuple[int, int, int, bool]]: A list of tuples, each containing the coordinates (x, y) of the top-left
                                          corner of the bounding box, the width (height) of the bounding box, and
                                          a boolean flag indicating whether the hand is a left hand (True) or a
                                          right hand (False).

    Notes:
        - The width and height of the bounding boxes are equal since the network requires squared input.
        - The minimum bounding box size is 20 pixels.
    """
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    left_shoulder = keypoints[5]
    left_elbow = keypoints[6]
    left_wrist = keypoints[7]
    right_shoulder = keypoints[2]
    right_elbow = keypoints[3]
    right_wrist = keypoints[4]

    # if any of three not detected
    has_left = all(keypoint is not None for keypoint in (left_shoulder, left_elbow, left_wrist))
    has_right = all(keypoint is not None for keypoint in (right_shoulder, right_elbow, right_wrist))
    if not (has_left or has_right):
        return []
    
    hands = []
    #left hand
    if has_left:
        hands.append([
            left_shoulder.x, left_shoulder.y,
            left_elbow.x, left_elbow.y,
            left_wrist.x, left_wrist.y,
            True
        ])
    # right hand
    if has_right:
        hands.append([
            right_shoulder.x, right_shoulder.y,
            right_elbow.x, right_elbow.y,
            right_wrist.x, right_wrist.y,
            False
        ])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
        # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
        # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
        # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
        # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
        # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        # x-y refers to the center --> offset to topLeft point
        # handRectangle.x -= handRectangle.width / 2.f;
        # handRectangle.y -= handRectangle.height / 2.f;
        x -= width / 2
        y -= width / 2  # width = height
        # overflow the image
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)
        # the max hand box value is 20 pixels
        if width >= 20:
            detect_result.append((int(x), int(y), int(width), is_left))

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# Written by Lvmin
def faceDetect(body: BodyResult, oriImg) -> Union[Tuple[int, int, int], None]:
    """
    Detect the face in the input body pose keypoints and calculate the bounding box for the face.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        Tuple[int, int, int] | None: A tuple containing the coordinates (x, y) of the top-left corner of the
                                   bounding box and the width (height) of the bounding box, or None if the
                                   face is not detected or the bounding box width is less than 20 pixels.

    Notes:
        - The width and height of the bounding box are equal.
        - The minimum bounding box size is 20 pixels.
    """
    # left right eye ear 14 15 16 17
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    head = keypoints[0]
    left_eye = keypoints[14]
    right_eye = keypoints[15]
    left_ear = keypoints[16]
    right_ear = keypoints[17]
    
    if head is None or all(keypoint is None for keypoint in (left_eye, right_eye, left_ear, right_ear)):
        return None

    width = 0.0
    x0, y0 = head.x, head.y

    if left_eye is not None:
        x1, y1 = left_eye.x, left_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if right_eye is not None:
        x1, y1 = right_eye.x, right_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if left_ear is not None:
        x1, y1 = left_ear.x, left_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    if right_ear is not None:
        x1, y1 = right_ear.x, right_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    x, y = x0, y0

    x -= width
    y -= width

    if x < 0:
        x = 0

    if y < 0:
        y = 0

    width1 = width * 2
    width2 = width * 2

    if x + width > image_width:
        width1 = image_width - x

    if y + width > image_height:
        width2 = image_height - y

    width = min(width1, width2)

    if width >= 20:
        return int(x), int(y), int(width)
    else:
        return None


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j