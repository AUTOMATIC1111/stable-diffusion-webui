# High Quality Edge Thinning using Pure Python
# Written by Lvmin Zhang
# 2023 April
# Stanford University
# If you use this, please Cite "High Quality Edge Thinning using Pure Python", Lvmin Zhang, In Mikubill/sd-webui-controlnet.


import cv2
import numpy as np


lvmin_kernels_raw = [
    np.array([
        [-1, -1, -1],
        [0, 1, 0],
        [1, 1, 1]
    ], dtype=np.int32),
    np.array([
        [0, -1, -1],
        [1, 1, -1],
        [0, 1, 0]
    ], dtype=np.int32)
]

lvmin_kernels = []
lvmin_kernels += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_kernels_raw]
lvmin_kernels += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_kernels_raw]

lvmin_prunings_raw = [
    np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [0, 0, -1]
    ], dtype=np.int32),
    np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 0, 0]
    ], dtype=np.int32)
]

lvmin_prunings = []
lvmin_prunings += [np.rot90(x, k=0, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=1, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=2, axes=(0, 1)) for x in lvmin_prunings_raw]
lvmin_prunings += [np.rot90(x, k=3, axes=(0, 1)) for x in lvmin_prunings_raw]


def remove_pattern(x, kernel):
    objects = cv2.morphologyEx(x, cv2.MORPH_HITMISS, kernel)
    objects = np.where(objects > 127)
    x[objects] = 0
    return x, objects[0].shape[0] > 0


def thin_one_time(x, kernels):
    y = x
    is_done = True
    for k in kernels:
        y, has_update = remove_pattern(y, k)
        if has_update:
            is_done = False
    return y, is_done


def lvmin_thin(x, prunings=True):
    y = x
    for i in range(32):
        y, is_done = thin_one_time(y, lvmin_kernels)
        if is_done:
            break
    if prunings:
        y, _ = thin_one_time(y, lvmin_prunings)
    return y


def nake_nms(x):
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    return y

