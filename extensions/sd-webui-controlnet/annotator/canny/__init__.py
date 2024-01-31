import cv2


def apply_canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
