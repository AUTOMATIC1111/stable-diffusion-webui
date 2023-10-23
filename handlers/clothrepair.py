#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 3:53 PM
# @Author  : wangdongming
# @Site    :
# @File    : gen_fusion_img.py
# @Software: Hifive
import random
import time
import typing
import modules
from modules import shared, scripts
from enum import IntEnum
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus

from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path, \
    save_processed_images_shanshu
from copy import deepcopy
import numpy as np
from typing import List, Union, Dict, Set, Tuple
import cv2
import numpy as np
from PIL import Image
import insightface
import onnxruntime
import os
import logging
from loguru import logger
import math

from handlers.clothes_repair.clothes_seg.clothes_seg import *
from PIL import ImageChops, Image, ImageDraw, ImageFilter, ImageOps
from handlers.clothes_repair.inpaint_fun.inpaint_clothes import *
from scipy.ndimage import binary_dilation, binary_erosion
from modules import script_callbacks, sd_models


def RGBA_to_other(image, option, blue=None, green=None, red=None):
    # 将图像转换为RGBA模式（如果尚未是RGBA模式）
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # 获取图像数据
    pixels = image.load()

    # 图像尺寸
    width, height = image.size

    # 遍历图像的每个像素
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]

            # 将RGB值与Alpha通道值求交集
            if a == 0:
                if option == "白色":
                    r = g = b = 255

                elif option == "黑色":
                    r = g = b = 0

                elif option == "灰黑色":
                    r = g = b = 50

                elif option == "灰白色":
                    r = g = b = 200

                elif option == "auto":
                    r = int(red)
                    g = int(green)
                    b = int(blue)

            # 更新像素值
            pixels[x, y] = (r, g, b, a)

    return image


def people_predict(input_image, text_prompt=None):
    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")
    print("Start SAM Processing")
    sam_model_name = sam_model_list[0]
    positive_points = []
    negative_points = []
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    if not text_prompt:
        text_prompt = "head, hair, brain"
    dino_enabled = text_prompt is not None
    box_threshold = 0.3
    dino_model_name = dino_model_list[0]
    boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)

    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    # if dino_enabled and boxes_filt.shape[0] > 1:
    is_rount = 1
    sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
    print(sam_predict_status)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()

    garbage_collect(sam)
    masks_gallery = []
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
    return masks_gallery


def clothes_predict(input_image):
    print("Start SAM Processing")
    sam_model_name = sam_model_list[0]
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]

    positive_points = [[290, 400], [425, 400], [285, 550], [425, 550], [360, 475]]
    negative_points = [[358, 324], [150, 485], [540, 467], [290, 710], [418, 710], [309, 243], [421, 247], [90, 177],
                       [600, 130], [295, 642], [425, 642]]
    text_prompt = None
    dino_enabled = text_prompt is not None
    boxes_filt = None
    box_threshold = 0.3
    sam_predict_result = " done."
    dino_model_name = dino_model_list[0]
    if dino_enabled:
        boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)

    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    if dino_enabled and boxes_filt.shape[0] > 1:
        sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
        print(sam_predict_status)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=False)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    else:
        num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if num_box == 0 and num_points == 0:
            garbage_collect(sam)
            return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
        sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, " \
                             f"{len(negative_points)} negative prompts"
        print(sam_predict_status)
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=False)
        masks = masks[:, None, ...]
    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt)


def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    weight = 2
    dilation_kernel = ((x - center) ** 2 + weight * (y - center) ** 2 <= center ** 2) | (y > center)
    dilation_kernel = dilation_kernel.astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask


def create_rectangle_mask(coord1, coord2, image_size):
    mask = Image.new('1', image_size, 0)
    draw = ImageDraw.Draw(mask)

    x1, y1 = coord1
    x2, y2 = coord2

    # 确定矩形的左上角和右下角坐标
    left_top = (min(x1, x2), min(y1, y2))
    right_bottom = (max(x1, x2), max(y1, y2))

    # 构建矩形蒙版区域
    draw.rectangle([left_top, right_bottom], fill=255)

    return mask


def resize_and_overlay_image(background_image, foreground_image, x, y, width, height):
    # 裁剪image_A的指定区域
    cropped_region = background_image.crop((x, y, x + width, y + height))

    # 将image_B resize为与裁剪的区域大小相同
    resized_image_B = foreground_image.resize((width, height))

    # 获取resized_image_B的A通道
    alpha = resized_image_B.split()[-1]

    # 将image_B的像素值替换为image_A对应的像素值
    image_B_pixels = resized_image_B.load()
    cropped_region_pixels = cropped_region.load()

    for i in range(width):
        for j in range(height):
            if alpha.getpixel((i, j)) != 255:
                cropped_region_pixel = tuple(cropped_region_pixels[i, j][:3])  # 提取前三个通道值
                image_B_pixels[i, j] = cropped_region_pixel + (cropped_region_pixels[i, j][-1],)

    # 创建一个新的空白图像，作为合并后的结果
    result_image = Image.new('RGBA', background_image.size)

    # 将image_A复制到result_image
    result_image.paste(background_image, (0, 0))

    # 将resize后的image_B覆盖到指定区域
    result_image.paste(resized_image_B, (x, y), mask=alpha)

    return result_image


def extract_alpha_channel(source_img, target_img):
    # Step 2: Extract alpha channel from the source image
    source_channels = source_img.split()  # Separates RGBA into individual channels
    alpha_channel = source_channels[3]

    # Step 3: Merge alpha channel with the target image's RGB channels
    target_channels = target_img.split()
    merged_img = Image.merge('RGBA', (target_channels[0], target_channels[1], target_channels[2], alpha_channel))

    return merged_img


def blend_images_with_mask(imageA, imageB, mask):
    # 将mask转换为灰度图像
    mask_gray = mask.convert("L")

    # 创建一个新的空白图像
    merged_image = Image.new("RGBA", mask.size)

    # 获取mask的像素数据
    mask_data = mask_gray.getdata()

    # 根据mask中像素值的不同，将imageA和imageB的像素值复制到新图像中
    for i, pixel_value in enumerate(mask_data):
        if pixel_value == 255:  # 白色区域
            merged_image.putpixel((i % mask.width, i // mask.width), imageA.getpixel((i % mask.width, i // mask.width)))
        else:  # 黑色区域
            merged_image.putpixel((i % mask.width, i // mask.width), imageB.getpixel((i % mask.width, i // mask.width)))

    return merged_image


def resize_image(image, target_size):
    # 调整图片尺寸并填充
    resized_image = ImageOps.pad(image, target_size)

    return resized_image


def crop_image(image, top_left, bottom_right):
    return image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))


def rotate_image(image, angle):
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)


def crop_image_vertically(image, clothes_type):
    # 获取图片尺寸
    width, height = image.size

    # 计算要裁剪的高度
    cropped_height = height // 4
    if clothes_type == "cardigan":
        cropped_height = height // 4
    elif clothes_type == "hoodie":
        cropped_height = height // 3

    # 裁剪图片
    cropped_image = image.crop((0, cropped_height, width, height))

    return cropped_image


def get_hoodie_image(imageA, imageB, coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10,
                     angle, clothes_type):
    # Define coordinates for areaB in imageB
    areaB_top_left = coord1
    areaB_bottom_right = coord2

    # Crop out areaB from imageB
    areaB = crop_image(imageB, areaB_top_left, areaB_bottom_right)

    # Define coordinates for areaC in imageB
    areaC_top_left = coord3
    areaC_bottom_right = coord4

    # Crop out areaC from imageB
    areaC = crop_image(areaB, areaC_top_left, areaC_bottom_right)
    # Get symmetric areaD of areaC
    areaD = crop_image(areaB, (imageB.width - areaC.width, 0), (imageB.width, areaC.height))

    areaA = crop_image(imageB, (0, 0), (areaB_bottom_right[0], areaB_top_left[1]))

    imageB_core = crop_image(areaB, (areaC.width, 0), (imageB.width - areaD.width, areaC.height))
    # Rotate areaC and areaD with preserving alpha channel
    rotated_areaC = rotate_image(areaC, -angle)
    rotated_areaD = rotate_image(areaD, angle)

    rotated_areaC = crop_image_vertically(rotated_areaC, clothes_type)
    rotated_areaD = crop_image_vertically(rotated_areaD, clothes_type)

    res_img = resize_and_overlay_image(imageA, rotated_areaC, coord5[0], coord5[1], coord6[0] - coord5[0],
                                       coord6[1] - coord5[1])
    res_img = resize_and_overlay_image(res_img, rotated_areaD, coord7[0], coord7[1], coord8[0] - coord7[0],
                                       coord8[1] - coord7[1])
    res_img = resize_and_overlay_image(res_img, imageB_core, coord9[0], coord9[1], coord10[0] - coord9[0],
                                       coord10[1] - coord9[1])

    if clothes_type == "cardigan":
        res_img = resize_and_overlay_image(res_img, areaA, 237, 312, 235, 35)

    return res_img


def get_coord(stable_area, clothes_type, frontor_back):
    if clothes_type == "shirt":
        if frontor_back == "正面":
            # 示例坐标点  #正面 T-shirt, coord1,coord2确定覆盖到模板图像的区域， coord3，coord4确定不重绘的区域
            coord1, coord2 = (185, 332), (529, 602)
            coord3, coord4 = (269, 377), (441, 574)
            if stable_area is not None and stable_area != "":
                coord3, coord4 = stable_area[0], stable_area[1]
        else:
            # 示例坐标点  #背面 T-shirt
            coord1, coord2 = (192, 333), (518, 591)
            coord3, coord4 = (276, 356), (443, 559)
            if stable_area is not None and stable_area != "":
                coord3, coord4 = stable_area[0], stable_area[1]
        return coord1, coord2, coord3, coord4

    if clothes_type == "cardigan":
        if frontor_back == "正面":
            # coord1~coord4裁剪和旋转服装图像， coord5~coord10确定覆盖到模板图像的区域
            coord1, coord2, coord3, coord4 = (0, 118), (900, 1104), (0, 0), (152, 986)
            coord5, coord6, coord7, coord8, coord9, coord10 = (130, 327), (273, 610), (447, 327), (590, 610), (
            243, 326), (467, 636)
            coord11, coord12 = (260, 358), (450, 587)
            if stable_area is not None and stable_area != "":
                coord11, coord12 = stable_area[0], stable_area[1]
            return coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10, coord11, coord12
        else:
            coord1, coord2, coord3, coord4 = (0, 118), (900, 1104), (0, 0), (152, 986)
            coord5, coord6, coord7, coord8, coord9, coord10 = (130, 327), (273, 610), (447, 327), (590, 610), (
                243, 326), (467, 636)
            coord11, coord12 = (282, 384), (429, 570)
            if stable_area is not None and stable_area != "":
                coord11, coord12 = stable_area[0], stable_area[1]
            return coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10, coord11, coord12

    if clothes_type == "hoodie":
        if frontor_back == "正面":
            # coord1~coord4裁剪和旋转服装图像， coord5~coord10确定覆盖到模板图像的区域
            coord1, coord2, coord3, coord4 = (0, 191), (779, 815), (0, 0), (149, 624)
            coord5, coord6, coord7, coord8, coord9, coord10 = (130, 327), (273, 610), (447, 327), (590, 610), (
                230, 326), (480, 636)
            coord11, coord12 = (281, 392), (425, 590)
            if stable_area is not None and stable_area != "":
                coord11, coord12 = stable_area[0], stable_area[1]
            return coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10, coord11, coord12
        else:
            coord1, coord2, coord3, coord4 = (0, 191), (779, 815), (0, 0), (149, 624)
            coord5, coord6, coord7, coord8, coord9, coord10 = (130, 327), (273, 610), (447, 327), (590, 610), (
                230, 326), (480, 636)
            coord11, coord12 = (285, 385), (438, 563)
            if stable_area is not None and stable_area != "":
                coord11, coord12 = stable_area[0], stable_area[1]
            return coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10, coord11, coord12


class ClothesRepairTask(Txt2ImgTask):
    def __init__(self,
                 base_model_path: str,  # 模型路径
                 model_hash: str,  # 模型hash值
                 clothes_img: str,  # 服装覆盖角色图
                 text_prompt: str,  # 图片的正向提示词
                 bg_color: str,  # 背景色
                 batch_size: int,  # 出图批数
                 clothes_type: str,  # 服装种类
                 frontor_back: str,  # 正反面
                 gender: str,  # 性别
                 is_psed: bool,  # 是否是ps过的图片
                 init_img: str,  # 模板图像数组
                 ):
        self.base_model_path = base_model_path
        self.model_hash = model_hash
        self.clothes_img = clothes_img
        self.text_prompt = text_prompt
        self.bg_color = bg_color
        self.batch_size = batch_size
        self.clothes_type = clothes_type
        self.frontor_back = frontor_back
        self.gender = gender
        self.is_psed = is_psed
        self.init_img = init_img

    @classmethod
    def exec_task(cls, task: Task):
        t = ClothesRepairTask(
            base_model_path=task['base_model_path'],
            model_hash=task['model_hash'],
            clothes_img=task.get('clothes_img'),
            bg_color=task.get('bg_color'),
            batch_size=task.get('batch_size'),
            clothes_type=task.get('clothes_type'),
            frontor_back=task.get('frontor_back'),
            gender=task.get('gender'),
            text_prompt=task.get('text_prompt', ''),
            is_psed=task.get('is_psed'),
            init_img=task.get('init_img'))
        full_task = deepcopy(task)

        return full_task


class ClothesRepairTaskType(Txt2ImgTask):
    RepairAction = 1  # 服装修复


class ClothesRepairTaskHandler(Txt2ImgTaskHandler):
    def __init__(self):
        super(ClothesRepairTaskHandler, self).__init__()
        self.task_type = TaskType.ClothesRepair

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        # 根据任务的不同类型：执行不同的任务
        if task.minor_type == ClothesRepairTaskType.RepairAction:
            yield from self._exec_ClothesRepair(task)

    def _exec_ClothesRepair(self, task: Task) -> typing.Iterable[TaskProgress]:

        full_task = ClothesRepairTask.exec_task(task)
        full_task.clothes_img = task['clothes_img']
        full_task.bg_color = task['bg_color']
        full_task.batch_size = task['batch_size']
        full_task.clothes_type = task['clothes_type']
        full_task.frontor_back = task['frontor_back']
        full_task.gender = task['gender']
        # full_task.text_prompt = task.get('text_prompt', '')
        full_task.text_prompt = ""
        full_task.is_psed = task['is_psed']
        full_task.init_img = task['init_img']

        split_values = full_task.init_img.split(",")
        full_task.init_img = split_values

        # base_model = os.path.join("oss://", full_task.base_model_path)
        # model_checkpoint = sd_models.CheckpointInfo(base_model)
        # modules.sd_models.reload_model_weights(info=model_checkpoint)

        # task['alwayson_scripts'] = {'ControlNet': {'args': []}}
        progress = TaskProgress.new_ready(task, 'clothes_repair')
        yield progress

        base_model_path = self._get_local_checkpoint(full_task)
        load_sd_model_weights(base_model_path, full_task.model_hash)

        clothes_img_url = get_tmp_local_path(full_task.clothes_img)
        clothes_img = Image.open(clothes_img_url)

        template_image_url = get_tmp_local_path(full_task.init_img[0])
        template_image = Image.open(template_image_url)

        if full_task.clothes_type != "shirt":
            mask_url = get_tmp_local_path(full_task.init_img[1])
            mask = Image.open(mask_url).convert("L")
        stable_area = None

        progress = TaskProgress.new_running(task, 'clothes_repair running')
        yield progress
        if isinstance(stable_area, str) and stable_area != "":
            stable_area = eval(stable_area)

        # 根据服装种类确定坐标（覆盖到模板图像的坐标或处理服装的坐标）
        if full_task.clothes_type == "shirt":
            coord1, coord2, coord3, coord4 = get_coord(stable_area, full_task.clothes_type, full_task.frontor_back)
        elif full_task.clothes_type == "hoodie" or "cardigan":
            coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8, coord9, coord10, coord11, coord12 = get_coord(
                stable_area, full_task.clothes_type, full_task.frontor_back)
        progress.task_progress = 10
        yield progress
        # 女版小人背面特殊处理
        if full_task.gender == "girl" and full_task.frontor_back != "正面":
            sp_mask = people_predict(template_image)[0]

        dilation_amt = 60

        if full_task.is_psed:
            if full_task.clothes_type == "shirt":
                # 给衣服裁一块儿区域
                x1, y1, width, height = coord1[0], coord1[1], coord2[0] - coord1[0], coord2[1] - coord1[1]
                fusion_image = resize_and_overlay_image(template_image, clothes_img, x1, y1, width, height)
                mask_list = clothes_predict(fusion_image)
                mask = mask_list[1]
                dilated_mask = dilate_mask(mask, dilation_amt)
                dilated_mask_arr = np.array(dilated_mask)

                coord5, coord6 = (301, 545), (434, 587)
                tmp_mask = create_rectangle_mask(coord3, coord4, mask.size)

                tmp_mask2 = create_rectangle_mask(coord5, coord6, mask.size)
                tmp_mask = ImageChops.logical_or(tmp_mask, tmp_mask2)

                tmp_mask_arr = np.logical_and(dilated_mask_arr, np.logical_not(np.array(tmp_mask)))

            if full_task.clothes_type == "hoodie" or full_task.clothes_type == "cardigan":
                if full_task.clothes_type == "hoodie":
                    angle = 60
                    fusion_image = get_hoodie_image(template_image, clothes_img, coord1, coord2, coord3, coord4, coord5,
                                                    coord6,
                                                    coord7, coord8, coord9, coord10, angle, full_task.clothes_type)

                elif full_task.clothes_type == "cardigan":
                    angle = 81
                    fusion_image = get_hoodie_image(template_image, clothes_img, coord1, coord2, coord3, coord4, coord5,
                                                    coord6,
                                                    coord7, coord8, coord9, coord10, angle, full_task.clothes_type)

                # 创建矩形蒙版
                tmp_mask = create_rectangle_mask(coord11, coord12, mask.size)
                tmp_mask_arr = np.logical_and(mask, np.logical_not(np.array(tmp_mask)))

        else:
            if full_task.clothes_type == "shirt":
                mask_list = clothes_predict(clothes_img)
                mask = mask_list[1]
                dilated_mask = dilate_mask(mask, dilation_amt)
                dilated_mask_arr = np.array(dilated_mask)

                coord5, coord6 = (301, 545), (434, 587)
                tmp_mask = create_rectangle_mask(coord3, coord4, mask.size)

                tmp_mask2 = create_rectangle_mask(coord5, coord6, mask.size)
                tmp_mask = ImageChops.logical_or(tmp_mask, tmp_mask2)

                tmp_mask_arr = np.logical_and(dilated_mask_arr, np.logical_not(np.array(tmp_mask)))

            if full_task.clothes_type == "hoodie" or full_task.clothes_type == "cardigan":
                # 创建矩形蒙版
                tmp_mask = create_rectangle_mask(coord11, coord12, mask.size)
                tmp_mask_arr = np.logical_and(mask, np.logical_not(np.array(tmp_mask)))

        progress.task_progress = 20
        yield progress

        # 女孩背面特殊处理
        if full_task.gender == "girl" and full_task.frontor_back != "正面":
            tmp_mask_arr = np.logical_and(tmp_mask_arr, np.logical_not(np.array(sp_mask)))

        tmp_mask = Image.fromarray(tmp_mask_arr.astype(np.uint8) * 255)

        # 根据是否是模板类似图决定需要改变背景颜色的图
        if full_task.is_psed:
            clothes_img = fusion_image
        else:
            clothes_img = clothes_img

        progress.task_progress = 30
        yield progress

        if full_task.bg_color == "auto":
            # 获取袖子区域的坐标
            x1, y1 = 215, 400
            x2, y2 = 235, 420
            roi = clothes_img.crop((x1, y1, x2, y2))
            roi_array = np.array(roi)

            # 分离RGB通道
            r, g, b, a = roi_array[:, :, 0], roi_array[:, :, 1], roi_array[:, :, 2], roi_array[:, :, 3]

            # 计算RGB通道的平均值
            average_r = np.mean(r)
            average_g = np.mean(g)
            average_b = np.mean(b)
            clothes_img = RGBA_to_other(clothes_img, "auto", average_b, average_g, average_r)

        elif full_task.bg_color == "黑色" or "白色" or "灰黑色" or "灰白色":
            clothes_img = RGBA_to_other(clothes_img, full_task.bg_color)

        image_dic = {}

        image_dic["image"] = clothes_img
        image_dic["mask"] = tmp_mask

        # 第二部分，重绘服装边缘，生图
        processed, process_args = inpaint_clothes(image_dic, template_image, full_task.text_prompt,
                                                  full_task.batch_size)
        repaired_image = processed.images

        res_list = []

        if full_task.batch_size > 1:
            repaired_image = repaired_image[1:-3]
        else:
            repaired_image = repaired_image[:-3]

        progress.task_progress = 60
        yield progress

        # 第三部分， 将模板图的alpha通道赋予生好的图
        for i in repaired_image:
            res_img = extract_alpha_channel(template_image, i)

            if full_task.gender == "girl" and full_task.frontor_back != "正面":
                res_img = blend_images_with_mask(template_image, res_img, sp_mask)

            target_size = (829, 1273)
            res_img = resize_image(res_img, target_size)
            res_list.append(res_img)

        progress.task_progress = 95
        yield progress
        processed.images = res_list

        images = save_processed_images_shanshu(processed,
                                               process_args.outpath_samples,
                                               process_args.outpath_grids,
                                               process_args.outpath_scripts,
                                               task.id)

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed.all_seeds, processed.all_subseeds)
        yield progress
