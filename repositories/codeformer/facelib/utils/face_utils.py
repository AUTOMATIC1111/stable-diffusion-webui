import cv2
import numpy as np
import torch


def compute_increased_bbox(bbox, increase_area, preserve_aspect=True):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    if preserve_aspect:
        width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
        height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    else:
        width_increase = height_increase = increase_area
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    return (left, top, right, bot)


def get_valid_bboxes(bboxes, h, w):
    left = max(bboxes[0], 0)
    top = max(bboxes[1], 0)
    right = min(bboxes[2], w)
    bottom = min(bboxes[3], h)
    return (left, top, right, bottom)


def align_crop_face_landmarks(img,
                              landmarks,
                              output_size,
                              transform_size=None,
                              enable_padding=True,
                              return_inverse_affine=False,
                              shrink_ratio=(1, 1)):
    """Align and crop face with landmarks.

    The output_size and transform_size are based on width. The height is
    adjusted based on shrink_ratio_h/shring_ration_w.

    Modified from:
    https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    Args:
        img (Numpy array): Input image.
        landmarks (Numpy array): 5 or 68 or 98 landmarks.
        output_size (int): Output face size.
        transform_size (ing): Transform size. Usually the four time of
            output_size.
        enable_padding (float): Default: True.
        shrink_ratio (float | tuple[float] | list[float]): Shring the whole
            face for height and width (crop larger area). Default: (1, 1).

    Returns:
        (Numpy array): Cropped face.
    """
    lm_type = 'retinaface_5'  # Options: dlib_5, retinaface_5

    if isinstance(shrink_ratio, (float, int)):
        shrink_ratio = (shrink_ratio, shrink_ratio)
    if transform_size is None:
        transform_size = output_size * 4

    # Parse landmarks
    lm = np.array(landmarks)
    if lm.shape[0] == 5 and lm_type == 'retinaface_5':
        eye_left = lm[0]
        eye_right = lm[1]
        mouth_avg = (lm[3] + lm[4]) * 0.5
    elif lm.shape[0] == 5 and lm_type == 'dlib_5':
        lm_eye_left = lm[2:4]
        lm_eye_right = lm[0:2]
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        mouth_avg = lm[4]
    elif lm.shape[0] == 68:
        lm_eye_left = lm[36:42]
        lm_eye_right = lm[42:48]
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        mouth_avg = (lm[48] + lm[54]) * 0.5
    elif lm.shape[0] == 98:
        lm_eye_left = lm[60:68]
        lm_eye_right = lm[68:76]
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        mouth_avg = (lm[76] + lm[82]) * 0.5

    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = mouth_avg - eye_avg

    # Get the oriented crop rectangle
    # x: half width of the oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
    # norm with the hypotenuse: get the direction
    x /= np.hypot(*x)  # get the hypotenuse of a right triangle
    rect_scale = 1  # TODO: you can edit it to get larger rect
    x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
    # y: half height of the oriented crop rectangle
    y = np.flipud(x) * [-1, 1]

    x *= shrink_ratio[1]  # width
    y *= shrink_ratio[0]  # height

    # c: center
    c = eye_avg + eye_to_mouth * 0.1
    # quad: (left_top, left_bottom, right_bottom, right_top)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    # qsize: side length of the square
    qsize = np.hypot(*x) * 2

    quad_ori = np.copy(quad)
    # Shrink, for large face
    # TODO: do we really need shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        h, w = img.shape[0:2]
        rsize = (int(np.rint(float(w) / shrink)), int(np.rint(float(h) / shrink)))
        img = cv2.resize(img, rsize, interpolation=cv2.INTER_AREA)
        quad /= shrink
        qsize /= shrink

    # Crop
    h, w = img.shape[0:2]
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, w), min(crop[3] + border, h))
    if crop[2] - crop[0] < w or crop[3] - crop[1] < h:
        img = img[crop[1]:crop[3], crop[0]:crop[2], :]
        quad -= crop[0:2]

    # Pad
    # pad: (width_left, height_top, width_right, height_bottom)
    h, w = img.shape[0:2]
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - w + border, 0), max(pad[3] - h + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w = img.shape[0:2]
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                           np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1],
                                           np.float32(h - 1 - y) / pad[3]))
        blur = int(qsize * 0.02)
        if blur % 2 == 0:
            blur += 1
        blur_img = cv2.boxFilter(img, 0, ksize=(blur, blur))

        img = img.astype('float32')
        img += (blur_img - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.clip(img, 0, 255)  # float32, [0, 255]
        quad += pad[:2]

    # Transform use cv2
    h_ratio = shrink_ratio[0] / shrink_ratio[1]
    dst_h, dst_w = int(transform_size * h_ratio), transform_size
    template = np.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
    # use cv2.LMEDS method for the equivalence to skimage transform
    # ref: https://blog.csdn.net/yichxi/article/details/115827338
    affine_matrix = cv2.estimateAffinePartial2D(quad, template, method=cv2.LMEDS)[0]
    cropped_face = cv2.warpAffine(
        img, affine_matrix, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray

    if output_size < transform_size:
        cropped_face = cv2.resize(
            cropped_face, (output_size, int(output_size * h_ratio)), interpolation=cv2.INTER_LINEAR)

    if return_inverse_affine:
        dst_h, dst_w = int(output_size * h_ratio), output_size
        template = np.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        affine_matrix = cv2.estimateAffinePartial2D(
            quad_ori, np.array([[0, 0], [0, output_size], [dst_w, dst_h], [dst_w, 0]]), method=cv2.LMEDS)[0]
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
    else:
        inverse_affine = None
    return cropped_face, inverse_affine


def paste_face_back(img, face, inverse_affine):
    h, w = img.shape[0:2]
    face_h, face_w = face.shape[0:2]
    inv_restored = cv2.warpAffine(face, inverse_affine, (w, h))
    mask = np.ones((face_h, face_w, 3), dtype=np.float32)
    inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
    # remove the black borders
    inv_mask_erosion = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))
    inv_restored_remove_border = inv_mask_erosion * inv_restored
    total_face_area = np.sum(inv_mask_erosion) // 3
    # compute the fusion edge based on the area of face
    w_edge = int(total_face_area**0.5) // 20
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
    img = inv_soft_mask * inv_restored_remove_border + (1 - inv_soft_mask) * img
    # float32, [0, 255]
    return img


if __name__ == '__main__':
    import os

    from facelib.detection import init_detection_model
    from facelib.utils.face_restoration_helper import get_largest_face

    img_path = '/home/wxt/datasets/ffhq/ffhq_wild/00009.png'
    img_name = os.splitext(os.path.basename(img_path))[0]

    # initialize model
    det_net = init_detection_model('retinaface_resnet50', half=False)
    img_ori = cv2.imread(img_path)
    h, w = img_ori.shape[0:2]
    # if larger than 800, scale it
    scale = max(h / 800, w / 800)
    if scale > 1:
        img = cv2.resize(img_ori, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
    if scale > 1:
        bboxes *= scale  # the score is incorrect
    bboxes = get_largest_face(bboxes, h, w)[0]

    landmarks = np.array([[bboxes[i], bboxes[i + 1]] for i in range(5, 15, 2)])

    cropped_face, inverse_affine = align_crop_face_landmarks(
        img_ori,
        landmarks,
        output_size=512,
        transform_size=None,
        enable_padding=True,
        return_inverse_affine=True,
        shrink_ratio=(1, 1))

    cv2.imwrite(f'tmp/{img_name}_cropeed_face.png', cropped_face)
    img = paste_face_back(img_ori, cropped_face, inverse_affine)
    cv2.imwrite(f'tmp/{img_name}_back.png', img)
