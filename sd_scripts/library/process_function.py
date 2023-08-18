import os
from PIL import Image,ImageOps
import cv2
import numpy as np
import io
from super_upscaler.super_upscaler import upscaler
from library.seg_rem_local_v2 import MySeg

# model_p = "/data/longcheng/stable-diffusion-webui/models"

def filter_images_by_size(image_list):
    smaller_than_1000 = []
    larger_than_1000 = []

    for image in image_list:
        width, height = image.size

        if width < 1000 or height < 1000:
            smaller_than_1000.append(image)
        else:
            larger_than_1000.append(image)

    return smaller_than_1000, larger_than_1000


# 镜像
def mirror_images(in_image):
    # 镜像图像
    mirrored_image = in_image.transpose(Image.FLIP_LEFT_RIGHT)
    return mirrored_image


# 添加背景
def add_back(imq, bg):
    # 读取要粘贴的图片 RGBA模式
    width, height = imq.size
    # whiteFrame = 255 * np.zeros((height, width, 3), np.uint8)
    # img = Image.open(bac_path)
    # r, g, b, a = imq.split()    # img.paste(imq, (0, 0, width, height), mask=a)
    r, g, b, a = imq.split()
    bg.paste(imq, (0, 0, width, height), mask=a)
    return bg


# resize并填充
def oversize(image, resize_weight, resize_height):

    # 调整图片尺寸并填充
    resized_image = ImageOps.pad(image, (resize_weight, resize_height))

    return resized_image


# # 放大
def upscale_process(img, scale=2,model_p=""):
    # dic = {}
    # dic["image"] = img
    img = upscaler(img, upscale_by=scale, style_type=1, upscaler_2_visibility=0.3, swap=True,models_path=model_p)
    img = oversize(img, img.width, img.height)
    return img


# 旋转 cv2
def rotate_pil_image(pil_image):
    angle1 = 15
    angle2 = -15

    # 将PIL图像转换为NumPy数组
    pil_image_array = np.array(pil_image)

    # 获取图像尺寸
    height, width = pil_image_array.shape[:2]

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix1 = cv2.getRotationMatrix2D(center, angle1, 1.0)
    rotation_matrix2 = cv2.getRotationMatrix2D(center, angle2, 1.0)

    # 执行图像旋转，使用双线性插值
    rotated_image_array1 = cv2.warpAffine(pil_image_array, rotation_matrix1, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rotated_image_array2 = cv2.warpAffine(pil_image_array, rotation_matrix2, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 将旋转后的NumPy数组转换回PIL图像
    rotated_pil_image1 = Image.fromarray(rotated_image_array1)
    rotated_pil_image2 = Image.fromarray(rotated_image_array2)

    return [rotated_pil_image1, rotated_pil_image2]


# # 旋转 pil
# def rotate_images_with_white(in_image):
#     angle1 = -15
#     angle2 = 15

#     # angle = np.random.randint(-30, 31)
#     fillcolor = "white"
#     rotated_image1 = in_image.rotate(angle1, expand=True, fillcolor=fillcolor)
#     rotated_image2 = in_image.rotate(angle2, expand=True, fillcolor=fillcolor)
#     return [rotated_image1, rotated_image2]


# 四通道图像背景修复
def RGBA_image_BGrepair(image, color):
    # 将图像转换为RGBA模式
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
                r = g = b = color

            # 更新像素值
            pixels[x, y] = (r, g, b, a)
    return image


def soften_edges(image, mask, blur_pixe=(35, 35)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = cv2.GaussianBlur(mask, blur_pixe, 0)       # 进行高斯模糊
    mask = mask / 255

    bg = np.zeros(image.shape, dtype='uint8')
    bg = 255 - bg                              # 转换成白色背景
    img = image / 255
    bg = bg / 255

    out = bg * (1 - mask) + img * mask        # 根据比例混合
    out = (out * 255).astype('uint8')
    opencv_image = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(opencv_image)
    return pil_image


def load_seg_model(load_model,model_p):
    myseg = None
    if load_model==True:
        myseg = MySeg(models_path=model_p)
    return myseg


def seg_hed(image, myseg):
    text_prompt = "head"
    list_seg = []
    seg_image = myseg.seg(image, text_prompt=text_prompt)
    for i in seg_image:
        list_seg.append(i)
    return list_seg


def seg_body(image, myseg):
    text_prompt = "full body"
    list_seg = []
    seg_image = myseg.seg(image, text_prompt=text_prompt)
    for i in seg_image:
        list_seg.append(i)
    return list_seg


def get_image(image_list):
    res = []
    for img in image_list:
        if not isinstance(img, Image.Image):
            img = Image.open(os.path.abspath(img))
        res.append(img)
    return res
