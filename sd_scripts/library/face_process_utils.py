import copy
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from skimage import transform


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    """
    Inputs:
        image                   Input image
        retinaface_result       The detection results of retinaface
        crop_ratio              The proportion of facial clipping and expansion
        face_seg                Facial segmentation model
        mask_type               The type of facial segmentation methods, one is loop and the other is skin, and the result of facial segmentation is the skin or frame of the face

    Outputs:
        retinaface_box          After box amplification, the box relative to the original image
        retinaface_keypoints    Points relative to the original image
        retinaface_mask_pil     Segmentation Results
    """
    h, w, c = np.shape(image)
    if len(retinaface_result["boxes"]) != 0:
        retinaface_boxs = []
        retinaface_keypoints = []
        retinaface_mask_pils = []
        for index in range(len(retinaface_result["boxes"])):
            # Obtain the box of reinface and expand it
            retinaface_box = np.array(retinaface_result["boxes"][index])
            face_width = retinaface_box[2] - retinaface_box[0]
            face_height = retinaface_box[3] - retinaface_box[1]
            retinaface_box[0] = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
            retinaface_box[1] = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 2, 0, h - 1)
            retinaface_box[2] = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
            retinaface_box[3] = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 2, 0, h - 1)
            retinaface_box = np.array(retinaface_box, np.int32)
            retinaface_boxs.append(retinaface_box)

            # Detect key points
            retinaface_keypoint = np.reshape(retinaface_result["keypoints"][index], [5, 2])
            retinaface_keypoint = np.array(retinaface_keypoint, np.float32)
            retinaface_keypoints.append(retinaface_keypoint)

            # mask part
            retinaface_crop = image.crop(np.int32(retinaface_box))
            retinaface_mask = np.zeros_like(np.array(image, np.uint8))
            if mask_type == "skin":
                retinaface_sub_mask = face_seg(retinaface_crop)
                retinaface_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = np.expand_dims(
                    retinaface_sub_mask, -1
                )
            else:
                retinaface_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 255
            retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
            retinaface_mask_pils.append(retinaface_mask_pil)

        retinaface_boxs = np.array(retinaface_boxs)
        argindex = np.argsort(retinaface_boxs[:, 0])
        retinaface_boxs = [retinaface_boxs[index] for index in argindex]
        retinaface_keypoints = [retinaface_keypoints[index] for index in argindex]
        retinaface_mask_pils = [retinaface_mask_pils[index] for index in argindex]
        return retinaface_boxs, retinaface_keypoints, retinaface_mask_pils

    else:
        retinaface_box = np.array([])
        retinaface_keypoints = np.array([])
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))

        return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def safe_get_box_mask_keypoints_and_padding_image(image, retinaface_detection, crop_ratio):
    """
    Get the face bounding box and padding image safely.

    Args:
        image (PIL.Image.Image): Input image.
        retinaface_detection: RetinaFace detection model.
        crop_ratio (float): expand ratio of face for padding.

    Returns:
        image (PIL.Image.Image): Padded image.
        retinaface_box (numpy.ndarray): Bounding box coordinates.
        retinaface_keypoint (numpy.ndarray): Key points coordinates.
        retinaface_mask_pil (PIL.Image.Image): Mask image.
        padding_size (int): Padding size.

    """
    h, w, c = np.shape(image)
    retinaface_result = retinaface_detection(image)

    # Check if face is detected
    if len(retinaface_result["boxes"]) == 0:
        return None, None, None, None, None
    retinaface_box = np.array(retinaface_result["boxes"][0])

    middle_x = (retinaface_box[0] + retinaface_box[2]) / 2
    middle_y = (retinaface_box[1] + retinaface_box[3]) / 2
    width = retinaface_box[2] - retinaface_box[0]
    height = retinaface_box[3] - retinaface_box[1]
    border_length = max(width, height)

    left = middle_x - border_length * (crop_ratio - 1) / 2
    top = middle_y - border_length * (crop_ratio - 1) / 2
    right = left + border_length * (crop_ratio - 1)
    bottom = top + border_length * (crop_ratio - 1)

    # Padding border for crop
    padding_left = -left
    padding_top = -top
    padding_right = -(w - right)
    padding_bottom = -(h - bottom)
    padding_size = int(max([padding_left, padding_top, padding_right, padding_bottom]) + 20)

    # Pad the image with white pixels
    if padding_size >= 0:
        image = np.pad(
            np.array(image, np.uint8),
            [[int(padding_size), int(padding_size)], [int(padding_size), int(padding_size)], [0, 0]],
            constant_values=255,
            mode="constant",
        )
    image = Image.fromarray(np.uint8(image))

    # Detect key points
    retinaface_keypoint = np.reshape(retinaface_result["keypoints"][0], [5, 2])
    retinaface_keypoint = np.array(retinaface_keypoint, np.float32)
    retinaface_keypoint = retinaface_keypoint + padding_size

    retinaface_box = np.array([left, top, right, bottom], np.int32) + padding_size

    # Create mask image
    retinaface_mask = np.zeros_like(np.array(image, np.uint8))
    retinaface_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 255
    retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))

    return image, retinaface_box, retinaface_keypoint, retinaface_mask_pil, padding_size


def crop_and_paste(source_image, source_image_mask, target_image, source_five_point, target_five_point, source_box):
    """
    Applies a face replacement by cropping and pasting one face onto another image.

    Args:
        source_image (PIL.Image): The source image containing the face to be pasted.
        source_image_mask (PIL.Image): The mask representing the face in the source image.
        target_image (PIL.Image): The target image where the face will be pasted.
        source_five_point (numpy.ndarray): Five key points of the face in the source image.
        target_five_point (numpy.ndarray): Five key points of the corresponding face in the target image.
        source_box (list): Coordinates of the bounding box around the face in the source image.

    Returns:
        PIL.Image: The resulting image with the pasted face.

    Notes:
        The function takes a source image, its corresponding mask, a target image, key points, and the bounding box
        around the face in the source image. It then aligns and pastes the face from the source image onto the
        corresponding location in the target image, taking into account the key points and bounding box.
    """
    source_five_point = np.reshape(source_five_point, [5, 2]) - np.array(source_box[:2])
    target_five_point = np.reshape(target_five_point, [5, 2])

    crop_source_image = source_image.crop(np.int32(source_box))
    crop_source_image_mask = source_image_mask.crop(np.int32(source_box))
    source_five_point, target_five_point = np.array(source_five_point), np.array(target_five_point)

    tform = transform.SimilarityTransform()
    # The program directly estimates the transformation matrix M
    tform.estimate(source_five_point, target_five_point)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(np.array(crop_source_image), M, np.shape(target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(crop_source_image_mask), M, np.shape(target_image)[:2][::-1], borderValue=0.0)

    mask = np.float32(warped_mask == 0)
    output = mask * np.float32(target_image) + (1 - mask) * np.float32(warped)
    return output


def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    # retinaface detect
    retinaface_result = retinaface_detection(image)
    # get mask and keypoints
    retinaface_box, retinaface_keypoints, retinaface_mask_pil = safe_get_box_mask_keypoints(
        image, retinaface_result, crop_ratio, None, "crop"
    )

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def call_face_crop_templates(loop_template_image, retinaface_detection, crop_face_preprocess):
    """
    Args:
        loop_template_image (list): A list of template images.
        retinaface_detection: The retinaface detection model.
        crop_face_preprocess (bool): Whether to crop face in preprocessing.

    Returns:
        input_image: A tuple containing the cropped template images and
        loop_template_crop_safe_box: The corresponding safe crop boxes.
    """
    input_image = []
    loop_template_crop_safe_box = []

    if crop_face_preprocess:
        loop_template_retinaface_box = []
        for _loop_template_image in loop_template_image:
            _loop_template_retinaface_boxes, _, _ = call_face_crop(retinaface_detection, _loop_template_image, 3, "loop_template_image")
            if len(_loop_template_retinaface_boxes) == 0:
                continue
            _loop_template_retinaface_box = _loop_template_retinaface_boxes[0]
            loop_template_retinaface_box.append(_loop_template_retinaface_box)
        if len(loop_template_retinaface_box) == 0:
            raise ValueError("There is no face in video.")
        # Get the enclose box of all boxes
        loop_template_retinaface_box = np.array(loop_template_retinaface_box)
        loop_template_retinaface_box = [
            np.min(loop_template_retinaface_box[:, 0]),
            np.min(loop_template_retinaface_box[:, 1]),
            np.max(loop_template_retinaface_box[:, 2]),
            np.max(loop_template_retinaface_box[:, 3]),
        ]

    for _loop_template_image in loop_template_image:
        # Crop the template image to retain only the portion of the portrait
        if crop_face_preprocess:
            _loop_template_crop_safe_box = loop_template_retinaface_box
            _loop_template_image = copy.deepcopy(_loop_template_image).crop(_loop_template_crop_safe_box)
        else:
            _loop_template_crop_safe_box = None

        input_image.append(_loop_template_image)
        loop_template_crop_safe_box.append(_loop_template_crop_safe_box)
    return input_image, loop_template_crop_safe_box


def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def alignment_photo(img, landmark, borderValue=(255, 255, 255)):
    """
    Rotate and align the image.

    Args:
        img: Input image.
        landmark: Landmark points coordinates.
        borderValue: Border fill value, default is white (255, 255, 255).

    Returns:
        new_img: Rotated and aligned image.
        new_landmark: Rotated and aligned landmark points coordinates.
    """
    x = landmark[0, 0] - landmark[1, 0]
    y = landmark[0, 1] - landmark[1, 1]
    angle = 0 if x == 0 else math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]), borderValue=borderValue)

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)
    return new_img, new_landmark


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


# This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        # here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode="bilinear", align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class Face_Skin(object):
    """
    Inputs:
        image   input image.
    Outputs:
        mask    output mask.
    """

    def __init__(self, model_path) -> None:
        n_classes = 19
        self.model = BiSeNet(n_classes=n_classes)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        # transform for input image
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    # index => label
    # 1:'skin', 2:'left_brow', 3:'right_brow', 4:'left_eye', 5:'right_eye', 6:'eye_g', 7:'left_ear', 8:'right_ear',
    # 9:'ear_r', 10:'nose', 11:'mouth', 12:'upper_lip', 13:'low_lip', 14:'neck', 15:'neck_l', 16:'cloth',
    # 17:'hair', 18:'hat'
    def __call__(self, image, retinaface_detection, needs_index=[[12, 13]]):
        # needs_index 12, 13 means seg the lip
        with torch.no_grad():
            total_mask = np.zeros_like(np.uint8(image))

            # detect image
            retinaface_boxes, _, _ = call_face_crop(retinaface_detection, image, 1.5, prefix="tmp")
            if len(retinaface_boxes) > 0:
                retinaface_box = retinaface_boxes[0]
                # sub_face for seg skin
                sub_image = image.crop(retinaface_box)
            else:
                sub_image = image

            # sub_face for seg skin
            sub_image = image.crop(retinaface_box)

            image_h, image_w, c = np.shape(np.uint8(sub_image))
            PIL_img = Image.fromarray(np.uint8(sub_image))
            PIL_img = PIL_img.resize((512, 512), Image.BILINEAR)

            torch_img = self.trans(PIL_img)
            torch_img = torch.unsqueeze(torch_img, 0)
            if self.cuda:
                torch_img = torch_img.cuda()
            out = self.model(torch_img)[0]
            model_mask = out.squeeze(0).cpu().numpy().argmax(0)

            masks = []
            for _needs_index in needs_index:
                total_mask = np.zeros_like(np.uint8(image))
                sub_mask = np.zeros_like(model_mask)
                for index in _needs_index:
                    sub_mask += np.uint8(model_mask == index)

                sub_mask = np.clip(sub_mask, 0, 1) * 255
                sub_mask = np.tile(np.expand_dims(cv2.resize(np.uint8(sub_mask), (image_w, image_h)), -1), [1, 1, 3])
                total_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2], :] = sub_mask
                masks.append(Image.fromarray(np.uint8(total_mask)))

            return masks
