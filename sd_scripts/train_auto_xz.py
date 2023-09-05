import os  # ,sys,re,torch
import shutil

from PIL import Image, ImageOps
# import random
import time
import tqdm
from enum import Enum
import math
import tempfile
# import diffusers
# from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
# from diffusers import (
#     StableDiffusionPipeline,
#     DDPMScheduler,
#     EulerAncestralDiscreteScheduler,
#     DPMSolverMultistepScheduler,
#     DPMSolverSinglestepScheduler,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     DDIMScheduler,
#     EulerDiscreteScheduler,
#     HeunDiscreteScheduler,
#     KDPM2DiscreteScheduler,
#     KDPM2AncestralDiscreteScheduler,
# )
from accelerate import Accelerator
# from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from train_network_all_auto import train_with_params, train_callback

# from prompts_generate import txt2img_prompts
from finetune.tag_images_by_wd14_tagger import get_wd_tagger
from library import autocrop
from library.process_function import *
from sd_scripts.super_upscaler.super_upscaler import upscaler
from sd_scripts.library import train_util
import uuid
import argparse
from finetune.deepbooru import deepbooru
import sys

sys.path.append("PaddleSeg/contrib/PP-HumanSeg")

import sd_scripts.library.config_util as config_util
import sd_scripts.library.custom_train_functions as custom_train_functions

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


class PreprocessTxtAction(Enum):
    Prepend = 'prepend'
    Append = 'append'
    Copy = 'copy'
    Ignore = 'ignore'


class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    process_caption = False
    process_caption_deepbooru = False
    preprocess_txt_action = None
    filter_tags = ""
    additional_tags = ""

    def do_caption(self, existing_caption=None):
        '''
        忽略已存在TAG
        '''
        if not existing_caption:
            return True
        return self.preprocess_txt_action in [
            PreprocessTxtAction.Prepend, PreprocessTxtAction.Copy, PreprocessTxtAction.Append
        ]


def save_pic_with_caption(image, index, params: PreprocessParams, existing_caption=None):
    caption = ""

    # if params.process_caption and params.do_caption(existing_caption):
    #     caption += shared.interrogator.generate_caption(image)

    if params.process_caption_deepbooru and params.do_caption(existing_caption):
        if len(caption) > 0:
            caption += ", "
        caption += deepbooru.model.tag_multi(image, force_disable_ranks=False,
                                             interrogate_deepbooru_score_threshold=0.45, deepbooru_use_spaces=False,
                                             deepbooru_escape=False, deepbooru_sort_alpha=False,
                                             interrogate_return_ranks=False,
                                             deepbooru_filter_tags=params.filter_tags,
                                             addtional_tags=params.additional_tags)

    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.preprocess_txt_action == 'prepend' and existing_caption:
        caption = existing_caption + ' ' + caption
    elif params.preprocess_txt_action == 'append' and existing_caption:
        caption = caption + ' ' + existing_caption
    elif params.preprocess_txt_action == 'copy' and existing_caption:
        caption = existing_caption

    caption = caption.strip()

    if len(caption) > 0:
        with open(os.path.join(params.dstdir, f"{basename}.txt"), "w", encoding="utf8") as file:
            file.write(caption)

    params.subindex += 1


def save_pic(image, index, params, existing_caption):
    save_pic_with_caption(image, index, params, existing_caption=existing_caption)
    # filename_part = params.src
    # filename_part = os.path.splitext(filename_part)[0]
    # filename_part = os.path.basename(filename_part)

    # basename = f"{index:05}-{params.subindex}-{filename_part}"
    # image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.flip:
        save_pic_with_caption(ImageOps.mirror(image), index, params, existing_caption=existing_caption)
        # image = ImageOps.mirror(image)
        # image.save(os.path.join(params.dstdir, f"{basename}.png"))


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted


# not using torchvision.transforms.CenterCrop because it doesn't allow float regions
def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    return wh and center_crop(image, *wh)


class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    process_caption = False
    process_caption_deepbooru = False
    preprocess_txt_action = None
    filter_tags = ""
    additional_tags = ""

    def do_caption(self, existing_caption=None):
        '''
        忽略已存在TAG
        '''
        if not existing_caption:
            return True
        return self.preprocess_txt_action in [
            PreprocessTxtAction.Prepend, PreprocessTxtAction.Copy, PreprocessTxtAction.Append
        ]


def save_pic_with_caption(image, index, params: PreprocessParams, existing_caption=None):
    caption = ""

    # if params.process_caption:
    #     caption += shared.interrogator.generate_caption(image)

    if params.process_caption_deepbooru:
        if len(caption) > 0:
            caption += ", "
        caption += deepbooru.model.tag_multi(image)

    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.preprocess_txt_action == 'prepend' and existing_caption:
        caption = existing_caption + ' ' + caption
    elif params.preprocess_txt_action == 'append' and existing_caption:
        caption = caption + ' ' + existing_caption
    elif params.preprocess_txt_action == 'copy' and existing_caption:
        caption = existing_caption

    caption = caption.strip()

    if len(caption) > 0:
        with open(os.path.join(params.dstdir, f"{basename}.txt"), "w", encoding="utf8") as file:
            file.write(caption)

    params.subindex += 1


def image_process(proc_image_input_batch, options, resize_weight=512, resize_height=512, if_res_oribody=False,
                  model_p=""):
    op1, op2, op3, op4, op5, op6 = "抠出头部", "抠出全身", "放大", "镜像", "旋转", "改变尺寸"

    if op1 or op2 in options:
        load_model = True

    myseg = load_seg_model(load_model, model_p)

    sum_list = []
    sum_head_image_list = []

    for image in proc_image_input_batch:
        image_process_list = []
        head_image_list = []

        image_process_list.append(image)
        if op1 in options:
            result_list = []
            for i in image_process_list:
                seg_head_img = seg_hed(i, myseg)
                # for j in range(len(seg_head_img)):
                #     # result_list.append(RGBA_image_BGrepair(seg_head_img[j], 255))
                #     result_list.append(seg_head_img)
                result_list += seg_head_img

            head_image_list += result_list

        if op2 in options:
            result_list = []
            for i in image_process_list:
                seg_body_img = seg_body(i, myseg)
                # for j in range(len(seg_body_img)):
                #     result_list.append(RGBA_image_BGrepair(seg_body_img[j], 255))

                result_list += seg_body_img
            # for img_tmp in result_list:
            #     path = add_suffix(os.path.abspath(img.name), '_body')
            #     img_tmp.save(path)

            if if_res_oribody:
                image_process_list += result_list
            else:
                image_process_list = result_list

        if op3 in options:
            result_list = []
            for i in image_process_list:
                if i.width * i.height >= 1024 * 1024:
                    result_list.append(i)
                else:
                    result_list.append(upscale_process(img=i, model_p=model_p))
            image_process_list = result_list

            result_list = []
            for i in head_image_list:
                if i.width * i.height >= 512 ** 2:
                    result_list.append(i)
                else:
                    result_list.append(upscale_process(img=i, model_p=model_p))
            head_image_list = result_list

        if op4 in options:
            result_list = []
            for i in image_process_list:
                result_list.append(mirror_images(i))
            image_process_list += result_list

            result_list = []
            for i in head_image_list:
                result_list.append(mirror_images(i))
            head_image_list += result_list

        if op5 in options:
            result_list = []
            for i in image_process_list:
                result_list += rotate_pil_image(i)
            image_process_list += result_list

            result_list = []
            for i in head_image_list:
                result_list += rotate_pil_image(i)
            head_image_list += result_list

        if op6 in options:
            result_list = []
            for i in image_process_list:
                result_list.append(oversize(i, resize_weight, resize_height))
            image_process_list = result_list

            result_list = []
            for i in head_image_list:
                result_list.append(oversize(i, resize_weight, resize_height))
            head_image_list = result_list

        sum_list += image_process_list
        sum_head_image_list += head_image_list

    # type_alone = False if tab_proc_index != 0 else True
    # res_path = save_imgprocess_batch(sum_list, proc_image_input_batch)

    myseg.stop()
    return sum_list, sum_head_image_list


def get_image_list(directory):
    image_list = []

    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否是图像文件（可根据需要添加其他文件格式的检查）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image_list.append(image_path)

    return image_list


def custom_configurable_image_processing(directory, options, resize_weight, resize_height, if_res_oribody, model_p):
    image_list = get_image_list(directory)
    images = get_image(image_list)
    # proc_image_input_batch = []
    # for img in images:
    #         image = RGBA_image_BGrepair(img, 255)
    #         proc_image_input_batch.append(image)

    res_list, head_list = image_process(images, options, resize_weight, resize_height, if_res_oribody, model_p)

    return res_list, head_list


def save_images_to_temp(images):
    temp_dir = tempfile.mkdtemp()  # 创建临时目录
    temp_image_paths = []

    for image in images:
        temp_image_filename = f"temp_image_{uuid.uuid4().hex}.png"  # 使用uuid生成随机文件名
        temp_image_path = f"{temp_dir}/{temp_image_filename}"  # 构造完整的临时文件路径
        image.save(temp_image_path)  # 保存图像到临时路径
        temp_image_paths.append(temp_image_path)

    return temp_dir, temp_image_paths


# 一键图片预处理函数
def train_preprocess(process_src, process_dst, process_width, process_height, preprocess_txt_action,
                     process_keep_original_size,
                     process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5,
                     overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
                     process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                     process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                     process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                     process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None,
                     model_path="",
                     filter_tags="", additional_tags=""):
    if process_caption_deepbooru:
        deepbooru.model.start()

    temp_dir, temp_image_paths = save_images_to_temp(process_src)
    width = process_width
    height = process_height
    src = os.path.abspath(temp_dir)
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    # assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)  # listfiles(src)

    params = PreprocessParams()
    params.dstdir = dst
    params.flip = process_flip
    params.process_caption = process_caption
    params.process_caption_deepbooru = process_caption_deepbooru
    params.preprocess_txt_action = preprocess_txt_action
    params.filter_tags = filter_tags
    params.additional_tags = additional_tags

    pbar = tqdm.tqdm(files)
    for index, imagefile in enumerate(pbar):
        # params.subindex = 0
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception:
            continue

        description = f"Preprocessing [Image {index}/{len(files)}]"
        pbar.set_description(description)
        # shared.state.textinfo = description

        params.src = filename

        existing_caption = None
        existing_caption_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(existing_caption_filename):
            with open(existing_caption_filename, 'r', encoding="utf8") as file:
                existing_caption = file.read()

        # if shared.state.interrupted:
        #     break
        if img.width < 512 and img.height < 512:  # 对于尺寸不够的图片进行upscale
            ratio = 2
            img = deepbooru.resize_image(1, img, int(img.width * ratio), int(img.height * ratio), upscaler_name="xyx",
                                         models_path=model_path)

        if img.height > img.width:
            ratio = (img.width * height) / (img.height * width)
            inverse_xy = False
        else:
            ratio = (img.height * width) / (img.width * height)
            inverse_xy = True

        process_default_resize = True

        if process_split and ratio < 1.0 and ratio <= split_threshold:
            for splitted in split_pic(img, inverse_xy, width, height, overlap_ratio):
                save_pic(splitted, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                dnn_model_path = autocrop.download_and_cache_models(os.path.join(model_path, "opencv"))
            except Exception as e:
                print(
                    "Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.",
                    e)

            autocrop_settings = autocrop.Settings(
                crop_width=width,
                crop_height=height,
                face_points_weight=process_focal_crop_face_weight,
                entropy_points_weight=process_focal_crop_entropy_weight,
                corner_points_weight=process_focal_crop_edges_weight,
                annotate_image=process_focal_crop_debug,
                dnn_model_path=dnn_model_path,
            )
            for focal in autocrop.crop_image(img, autocrop_settings):
                save_pic(focal, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_multicrop:
            cropped = multicrop_pic(img, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea,
                                    process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)
            if cropped is not None:
                save_pic(cropped, index, params, existing_caption=existing_caption)
            else:
                print(
                    f"skipped {img.width}x{img.height} image {filename} (can't find suitable size within error threshold)")
            process_default_resize = False

        if process_keep_original_size:
            if img.width > 2048 or img.height > 2048:  # 对于超大图片限制在2048的范围内
                if img.width > img.height:
                    ratio = 2048.0 / img.width
                else:
                    ratio = 2048.0 / img.height
                img = deepbooru.resize_image(1, img, int(img.width * ratio), int(img.height * ratio))
            save_pic(img, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_default_resize:
            img = deepbooru.resize_image(1, img, width, height)
            save_pic(img, index, params, existing_caption=existing_caption)

        # os.remove(filename)

        # shared.state.nextjob()
        if callable(progress_cb):
            current_progress = (index + 1) * 100 / len(files)
            if current_progress % 5 == 0:
                progress_cb(current_progress)
    for temp_path in temp_image_paths:
        os.remove(temp_path)
    if process_caption_deepbooru:
        deepbooru.model.stop()
    return


# 一键打标函数
def train_tagger(train_data_dir, model_dir, trigger_word=None, undesired_tags=None, general_threshold=0.35):
    get_wd_tagger(
        train_data_dir=train_data_dir,
        model_dir=model_dir,  # r"/data/qll/stable-diffusion-webui/models/tag_models",
        caption_extension=".txt",
        general_threshold=general_threshold,
        recursive=True,
        remove_underscore=True,
        undesired_tags=undesired_tags,
        additional_tags=trigger_word
    )
    return


MODEL_PATH = "/data/qll/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors"
PIC_SAVE_PATH = "/data/qll/pics/yijian_sanciyuan_train"
LORA_PATH = "/data/qll/stable-diffusion-webui/models/Lora"


def prepare_accelerator(logging_dir="./logs", log_prefix=None, gradient_accumulation_steps=1, mixed_precision="no"):
    if logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if log_prefix is None else log_prefix
        logging_dir = logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=None,
        project_dir=logging_dir,
    )

    # accelerateの互換性問題を解決する
    accelerator_0_15 = True
    try:
        accelerator.unwrap_model("dummy", True)
        print("Using accelerator 0.15.0 or above.")
    except TypeError:
        accelerator_0_15 = False

    def unwrap_model(model):
        if accelerator_0_15:
            return accelerator.unwrap_model(model, True)
        return accelerator.unwrap_model(model)

    return accelerator, unwrap_model


def train_callback(percentage):
    print(percentage)


# 一键训练入口函数
def train_auto(
        train_data_dir="",  # 训练的图片路径
        train_type=0,  # 训练的类别
        task_id=0,  # 任务id,作为Lora名称
        sd_model_path="",  # 底模路径
        lora_path="",  # 文件夹名字
        general_model_path="",  # 通用路径,
        train_callback=None,  # callback函数
        other_args=None  # 预留一个，以备用
):
    # 预设参数
    width_train = 512
    height_train = 768
    width = 512
    height = 768
    options = ["抠出头部", "放大"]  # 数据预处理方法 "抠出全身","抠出头部", "放大", "镜像", "旋转", "改变尺寸"
    head_width = 512
    head_height = 512
    trigger_word = ""
    # 是否采用wd14作为反推tag，否则采用deepbooru
    use_wd = os.getenv('WD', '1') == '1'

    # 反推tag默认排除的提示词
    undesired_tags = "blur,blurry,motion blur"  # 待测试五官

    # 图片处理后的路径
    dirname = os.path.dirname(train_data_dir)
    image_list, head_list = custom_configurable_image_processing(train_data_dir, options, width, height,
                                                                 if_res_oribody=True, model_p=general_model_path)
    train_dir = os.path.join(dirname, f"{task_id}-preprocess")
    os.makedirs(train_dir, exist_ok=True)
    process_dir = train_dir
    # print("1111:::", image_list, head_list)

    # 1.图片预处理
    train_preprocess(process_src=image_list, process_dst=train_dir, process_width=width, process_height=height,
                     preprocess_txt_action='ignore', process_keep_original_size=False,
                     process_split=False, process_flip=False, process_caption=True,
                     process_caption_deepbooru=not use_wd, split_threshold=0.5,
                     overlap_ratio=0.2, process_focal_crop=True, process_focal_crop_face_weight=0.9,
                     process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                     process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                     process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                     process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None,
                     model_path=general_model_path,
                     filter_tags=undesired_tags, additional_tags=trigger_word)
    train_callback(2)

    train_preprocess(process_src=head_list, process_dst=train_dir, process_width=head_width, process_height=head_height,
                     preprocess_txt_action='ignore', process_keep_original_size=False,
                     process_split=False, process_flip=False, process_caption=True,
                     process_caption_deepbooru=not use_wd, split_threshold=0.5,
                     overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
                     process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                     process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                     process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                     process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None,
                     model_path=general_model_path,
                     filter_tags=undesired_tags, additional_tags=trigger_word)
    train_callback(2)

    if use_wd and os.getenv("DO_NOT_TRAIN_COPY_ORIGIN", "1") != "1":
        for f in os.listdir(train_dir):
            full = os.path.join(train_dir, f)
            if os.path.isfile(full):
                target = os.path.join(process_dir, os.path.basename(f))
                shutil.copy(full, target)
    # # 优化图片数量
    contents = os.listdir(train_dir)
    if use_wd:
        pic_nums = len(contents)

    else:
        pic_nums = len(contents) / 2

    max_repeats = 40
    repeats_n = min(int(20 * max_repeats / pic_nums), max_repeats)

    # 2.tagger反推
    if use_wd:
        onnx = os.path.join(general_model_path, "tag_models/wd_onnx")
        if not os.path.isdir(onnx):
            raise OSError(f'cannot found onnx directory:{onnx}')

        train_tagger(
            train_data_dir=train_dir,
            model_dir=os.path.join(general_model_path, "tag_models/wd_onnx"),
            # r"/data/qll/stable-diffusion-webui/models/tag_models",
            general_threshold=0.35,
            undesired_tags=undesired_tags,
            trigger_word=trigger_word
        )

        if callable(train_callback):
            train_callback(2)

    lora_name = f"{task_id}"

    # 判断性别
    girl_count, man_count = 0, 0
    for f in os.listdir(process_dir):
        full = os.path.join(process_dir, f)
        _, ex = os.path.splitext(f)
        if ex.lower().lstrip('.') == 'txt':
            with open(full) as f:
                tags = ' '.join(f.readlines()).lower()
                if 'girl' in tags:
                    girl_count += 1
                elif 'boy' in tags:
                    man_count += 1

    gender = '1girl' if girl_count > man_count else '1boy'
    print(f">>>>>> gender:{gender}")

    # 3.自动训练出图
    train_with_params(
        pretrained_model_name_or_path=sd_model_path,
        network_weights=None,
        output_name=lora_name,  # f"num_r{num_r}_epo{epo}_net{net}_lr{al}",
        output_dir=lora_path,
        save_every_n_epochs=None,
        save_last_n_epochs=1,
        trigger_words=[""],  # [f"{task_id}",f"{task_id}"],
        list_train_data_dir=[process_dir],
        save_model_as="safetensors",
        num_repeats=[f"{repeats_n}"],
        batch_size=24,
        resolution=f"{width_train},{height_train}",
        epoch=10,  # 整数，随便填
        network_module="networks.lora",
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
        network_dim=32,  # 4的倍数，<=256
        network_alpha=32,  # 小于等于network_dim,可以不是4的倍数
        clip_skip=2,  # 0-12
        enable_preview=False,  # 和下面这几个参数一起的
        sample_prompts="",  # 文件路径，比如c:\promts.txt,file for prompts to generate sample images
        sample_sampler="euler_a",
        optimizer_type="AdamW8bit",
        learning_rate=0.0001,
        unet_lr=0.0001,
        text_encoder_lr=0.00001,
        lr_scheduler="cosine_with_restarts",
        auto_lr=True,
        auto_lr_param=6,

        cache_latents=True,
        # cache latents to main memory to reduce VRAM usage (augmentations must be disabled)
        cache_latents_to_disk=False,
        # cache latents to disk to reduce VRAM usage (augmentations must be disabled)
        enable_bucket=True,  # enable buckets for multi aspect ratio training
        min_bucket_reso=256,  # 范围自己定，minimum resolution for buckets
        max_bucket_reso=2048,  # 范围自己定，maximum resolution for buckets
        bucket_reso_steps=64,  # 秋叶版没有这个,steps of resolution for buckets, divisible by 8 is recommended
        bucket_no_upscale=True,  # 秋叶版没有这个,make bucket for each image without upscaling
        token_warmup_min=1,  # 秋叶版没有这个,start learning at N tags (token means comma separated strinfloatgs)
        token_warmup_step=0,  # 秋叶版没有这个,tag length reaches maximum on N steps (or N*max_train_steps if N<1)

        caption_extension=".txt",
        caption_dropout_rate=0.0,  # Rate out dropout caption(0.0~1.0)
        caption_dropout_every_n_epochs=0,  # Dropout all captions every N epochs
        caption_tag_dropout_rate=0.0,  # Rate out dropout comma separated tokens(0.0~1.0)
        shuffle_caption=False,  # shuffle comma-separated caption
        weighted_captions=False,  # 使用带权重的 token，不推荐与 shuffle_caption 一同开启
        keep_tokens=0,
        # keep heading N tokens when shuffling caption tokens (token means comma separated strings)
        color_aug=False,  # 秋叶版没有这个,enable weak color augmentation
        flip_aug=False,  # 秋叶版没有这个,enable horizontal flip augmentation
        face_crop_aug_range=None,  # 1.0,2.0,4.0
        # 秋叶版没有这个,enable face-centered crop augmentation and its range (e.g. 2.0,4.0)
        random_crop=False,
        # 秋叶版没有这个,enable random crop (for style training in face-centered crop augmentation)

        lowram=False,
        # enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)
        mem_eff_attn=False,  # use memory efficient attention for CrossAttention
        xformers=True,  # 如果mem_eff_attn为True则xformers设置无效
        vae=None,  # 比如：c:\vae.pt, 秋叶版没有这个,path to checkpoint of vae to replace
        max_data_loader_n_workers=8,
        # 秋叶版没有这个,max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)
        persistent_data_loader_workers=True,
        # persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory)

        max_train_steps=1600,  # 秋叶版没有这个,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        # 整数，随便填, Number of updates steps to accumulate before performing a backward/update pass
        mixed_precision="no",  # 是否使用混精度
        full_fp16=True,  # fp16 training including gradients

        # ["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",]
        sample_every_n_epochs=None,  # 1,2,3,4,5.....

        # network额外参数
        conv_dim=None,  # 默认为None，可以填4的倍数，类似于network_dim,
        # lycoris才有，# 4的倍数, 适用于lora,dylora。如果是dylora,则"conv_dim must be same as network_dim",
        conv_alpha=None,  # 默认为None，可以填比con_dim小的整数，类似于network_alpha； lycoris才有，<=conv_dim, 适用于lora,dylora
        unit=8,  # 秋叶版没有
        dropout=0,  # dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3暂时不支持
        algo='lora',  # 可选['lora','loha','lokr','ia3']

        enable_block_weights=False,  # 让下面几个参数有效
        block_dims=None,  # lora,  类似于network_dim,
        block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
        conv_block_dims=None,  # lora,  类似于network_dim,
        conv_block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
        down_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        mid_lr_weight=None,  # lora, 1位float,例如 1.0；
        up_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        block_lr_zero_threshold=0.0,  # float型，分层学习率置 0 阈值

        # AdamW (default), AdamW8bit, Lion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, AdaFactor
        weight_decay=0.01,  # optimizer_args,优化器内部的参数，权重衰减系数，不建议随便改
        betas=0.9,  # optimizer_args,优化器内部的参数，不建议随便改

        max_grad_norm=1.0,  # Max gradient norm, 0 for no clipping

        # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor
        lr_scheduler_num_cycles=1,  # Number of restarts for cosine scheduler with restarts
        lr_warmup_steps=0,  # Number of steps for the warmup in the lr scheduler
        lr_scheduler_power=1,  # Polynomial power for polynomial scheduler

        seed=918699631,
        prior_loss_weight=1.0,  # loss weight for regularization images
        min_snr_gamma=None,  # float型，比如5.0，最小信噪比伽马值，如果启用推荐为 5
        noise_offset=None,  # float型，0.1左右,enable noise offset with this value (if enabled, around 0.1 is recommended)
        adaptive_noise_scale=None,  # float型， 1.0
        # 与noise_offset配套使用；add `latent mean absolute value * this value` to noise_offset (disabled if None, default)
        multires_noise_iterations=6,  # 整数，多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。
        multires_noise_discount=0.3,  # 多分辨率（金字塔）噪声衰减率 推荐 6-10。无法与 noise_offset 一同启用。

        config_file=None,  # "test_config.toml",  # using .toml instead of args to pass hyperparameter
        output_config=False,  # output command line args to given .toml file
        # accelerator=accelerator,
        # unwrap_model=unwrap_model,
        callback=train_callback,
    )

    return os.path.join(lora_path, lora_name + ".safetensors"), gender
