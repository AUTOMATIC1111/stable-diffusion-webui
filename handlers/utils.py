#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 2:51 PM
# @Author  : wangdongming
# @Site    :
# @File    : utils.py
# @Software: Hifive
import io
import os
import time
import typing
import uuid
import hashlib
import cv2
import dlib
from PIL import Image
from loguru import logger
from datetime import datetime
from tools import TempDir as Tmp
from PIL.PngImagePlugin import PngInfo
from tools.wrapper import FuncExecTimeWrapper
from modules.shared import cmd_opts
from modules.processing import Processed
from modules.scripts import Script, ScriptRunner
from tools.environment import S3Tmp, S3SDWEB, enable_download_locker
from filestorage import FileStorageCls, get_local_path, batch_download
from handlers.formatter import format_alwayson_script_args, format_select_script_args
from handlers.typex import ModelLocation, ModelType, ImageOutput, OutImageType, UserModelLocation
from modules.sd_models import reload_model_weights, CheckpointInfo, get_closet_checkpoint_match, list_models

StrMapMap = typing.Dict[str, typing.Mapping[str, typing.Any]]


def detect_faces(image_path) -> int:
    # 加载图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1)
    logger.debug(f"image face detector:{len(faces)},{os.path.basename(image_path)}")
    return len(faces)


def get_model_local_path(remoting_path: str, model_type: ModelType, progress_callback=None):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path

    def ckpt_register(model_path: str):
        if model_type == ModelType.CheckPoint and os.path.isfile(model_path):
            checkpoint = CheckpointInfo(model_path)
            checkpoint.register()

    # 判断user-models下
    os.makedirs(UserModelLocation[model_type], exist_ok=True)
    dst = os.path.join(UserModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        ckpt_register(dst)
        return dst

    os.makedirs(ModelLocation[model_type], exist_ok=True)
    dst = os.path.join(ModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        ckpt_register(dst)
        return dst
    with_locker = enable_download_locker()
    dst = get_local_path(
        remoting_path, dst, progress_callback=progress_callback, locker_exp=1800, with_locker=with_locker)
    if os.path.isfile(dst):
        ckpt_register(dst)
        return dst


def batch_model_local_paths(model_type: ModelType, *remoting_paths: str) \
        -> typing.Sequence[str]:
    remoting_key_dst_pairs, loc = [], []
    os.makedirs(ModelLocation[model_type], exist_ok=True)
    for p in remoting_paths:
        if not p:
            continue
        if os.path.isfile(p):
            loc.append(p)
            continue

        dst = os.path.join(ModelLocation[model_type], os.path.basename(p))
        loc.append(dst)
        if not os.path.isfile(dst):
            remoting_key_dst_pairs.append((p, dst))
    batch_download(remoting_key_dst_pairs, with_locker=True)
    return loc


def mk_tmp_dir(dirname):
    d = os.path.join(Tmp, dirname)
    os.makedirs(d, exist_ok=True)
    return d


def get_tmp_local_path(remoting_path: str):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path

    os.makedirs(Tmp, exist_ok=True)
    _, ex = os.path.splitext(remoting_path)

    md5 = hashlib.md5()
    md5.update(remoting_path.encode())
    hash_str = md5.hexdigest()[:16]

    dst = os.path.join(Tmp, hash_str + ex)
    return get_local_path(remoting_path, dst, with_locker=False)


def upload_files(is_tmp, *files, dirname=None):
    keys = []
    if files:
        date = datetime.today().strftime('%Y/%m/%d')
        file_storage_system = FileStorageCls()
        relative = S3Tmp if is_tmp else S3SDWEB
        if dirname:
            relative = os.path.join(relative, dirname)

        for f in files:
            name = os.path.basename(f)
            key = os.path.join(relative, date, name)
            file_storage_system.upload(f, key)
            keys.append(key)
    return keys


def upload_content(is_tmp, content, name=None, dirname=None):
    date = datetime.today().strftime('%Y/%m/%d')

    file_storage_system = FileStorageCls()
    relative = S3Tmp if is_tmp else S3SDWEB
    if dirname:
        relative = os.path.join(relative, dirname)

    name = name or str(uuid.uuid1())
    key = os.path.join(relative, date, name)
    file_storage_system.upload_content(key, content)

    return key


def upload_pil_image(is_tmp, image, quality=80, name=None):
    with io.BytesIO() as output_bytes:
        use_metadata = False
        image.save(output_bytes, format="PNG", quality=quality)
        bytes_data = output_bytes.getvalue()
        return upload_content(is_tmp, bytes_data, name)


def strip_model_hash(model_name: str):
    if '[' not in model_name:
        return model_name
    start = model_name.rindex('[')
    if model_name[-1] == ']':
        return model_name[:start].strip()


def init_default_script_args(script_runner):
    # find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None] * last_arg_index
    script_args[0] = 0

    # get default values
    for script in script_runner.scripts:
        if script.args_from == script.args_to:
            continue

        script_args[script.args_from:script.args_to] = script.default_values

    return script_args


def get_script(script_name, script_runner):
    if script_name is None or script_name == "":
        return None

    script_idx = script_name_to_index(script_name, script_runner.scripts)
    return script_runner.scripts[script_idx]


def get_selectable_script(script_runner, script_name):
    if not script_name:
        return None, None

    script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
    script = script_runner.selectable_scripts[script_idx]
    return script, script_idx


def script_name_to_index(name, scripts):
    try:
        x = name.lower().replace(' ', '-')
        script_names = [script.title().lower().replace(' ', '-') for script in scripts]
        if x in script_names:
            return script_names.index(x)
        else:
            x = name.lower().replace("_", "-")  # for tile-diffusion
            return script_names.index(x)
    except:
        raise Exception(f"Script '{name}' not found")


ADetailer = 'ADetailer'
default_alwayson_scripts = {
    ADetailer: {
        'args': [{
            'ad_model': 'mediapipe_face_full'
        }]
    }
}


def init_script_args(default_script_args: typing.Sequence,
                     alwayson_scripts: StrMapMap,
                     selectable_scripts: Script,
                     selectable_idx: int,
                     request_script_args: typing.Union[typing.Sequence, typing.Mapping],
                     script_runner: ScriptRunner,
                     enable_def_adetailer: bool = True,
                     enable_refiner: bool = False,  # 是否启用XLRefiner
                     refiner_switch_at: float = 0.8,  # XL 精描切换时机
                     refiner_checkpoint: str = None,  # XL refiner模型文件
                     seed: int = -1,  # 随机种子
                     seed_enable_extras: bool = False,  # 是否启用随机种子扩展
                     subseed: int = -1,  # 差异随机种子
                     subseed_strength: float = 0,  # 差异强度
                     seed_resize_from_h: int = 0,  # 重置尺寸种子-高度
                     seed_resize_from_w: int = 0  # 重置尺寸种子-宽度
                     ):
    script_args = [x for x in default_script_args]

    if selectable_scripts:
        formatted_select_script_args = format_select_script_args(
            script_runner.is_img2img,
            selectable_scripts,
            request_script_args
        )

        script_args[selectable_scripts.args_from:selectable_scripts.args_to] = formatted_select_script_args
        script_args[0] = selectable_idx + 1

    alwayson_scripts = alwayson_scripts or {}
    alwayson_scripts.update({
        "Seed": {
            'args': [
                seed,
                seed_enable_extras,
                subseed,
                subseed_strength,
                seed_resize_from_h,
                seed_resize_from_w]
        }
    })

    if enable_refiner:
        if not os.path.isfile(refiner_checkpoint):
            raise ValueError('refiner_checkpoint not found')
        # processing.py中 sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        # 通过get_closet_checkpoint_match查找的模型，checkpoint_aliases需要提前注册（使用的文件名不含路径）
        # 而worker下面获取的basename作为hash, 因此refiner_checkpoint需要取basename
        basename = os.path.basename(refiner_checkpoint)
        name_for_extra, _ = os.path.splitext(basename)

        # 防止文件过期被删
        os.popen(f'touch {refiner_checkpoint}')
        checkpoint = CheckpointInfo(refiner_checkpoint)
        logger.debug(f"[XL REFINER] => register refiner model({name_for_extra}) ids:{checkpoint.ids}")
        checkpoint.register()

        # 检查是否能获取
        # if not get_closet_checkpoint_match(name_for_extra):
        #     logger.warning("cannot got refiner model, list models...")
        #     list_models()

        alwayson_scripts.update({
            "Refiner": {'args': [
                enable_refiner,
                name_for_extra,
                refiner_switch_at
            ]
            },
        })
    # 取消默认的AD
    # if not getattr(cmd_opts, 'disable_tss_def_alwayson', False):
    #     for k, v in default_alwayson_scripts.items():
    #         if not enable_def_adetailer and ADetailer == k:
    #             logger.debug('====> default adetailer plugin disable!')
    #             continue
    #         if k in alwayson_scripts:
    #             continue
    #
    #         alwayson_scripts[k] = v
    # else:
    #     logger.debug('====> disable tss default plugin!')

    # Now check for always on scripts
    if alwayson_scripts:

        for alwayson_script_name in alwayson_scripts.keys():
            alwayson_script = get_script(alwayson_script_name, script_runner)
            if not alwayson_script:
                raise Exception(f"always on script {alwayson_script_name} not found")
            # Selectable script in always on script param check
            if not alwayson_script.alwayson:
                raise Exception(f"Cannot have a selectable script in the always on scripts params")
            # always on script with no arg should always run so you don't really need to add them to the requests
            if "args" in (alwayson_scripts[alwayson_script_name] or {}):
                real_script_args = format_alwayson_script_args(alwayson_script_name,
                                                               script_runner.is_img2img,
                                                               alwayson_scripts[alwayson_script_name]["args"])
                real_script_arg_len = len(real_script_args)
                if real_script_arg_len != alwayson_script.args_to - alwayson_script.args_from:
                    expect = alwayson_script.args_to - alwayson_script.args_from
                    logger.warning(
                        f'[WARN] extension: {alwayson_script_name}, arguments unmatched, expect: {expect}, '
                        f'got:{real_script_arg_len}, front-end parameters may need to be updated.')

                script_args[alwayson_script.args_from: alwayson_script.args_from + real_script_arg_len] = \
                    real_script_args
            elif isinstance(alwayson_scripts[alwayson_script_name], (list, tuple)):
                real_script_args = format_alwayson_script_args(alwayson_script_name,
                                                               script_runner.is_img2img,
                                                               list(alwayson_scripts[alwayson_script_name]))
                real_script_arg_len = len(real_script_args)
                if real_script_arg_len != alwayson_script.args_to - alwayson_script.args_from:
                    expect = alwayson_script.args_to - alwayson_script.args_from
                    logger.warning(
                        f'[WARN] extension: {alwayson_script_name}, arguments unmatched, expect: {expect}, '
                        f'got:{real_script_arg_len}, front-end parameters may need to be updated.')

                script_args[alwayson_script.args_from: alwayson_script.args_from + real_script_arg_len] = \
                    real_script_args
    return script_args


@FuncExecTimeWrapper()
def load_sd_model_weights(filename, sha256=None):
    # 修改文件mtime，便于后续清理
    if filename:
        os.popen(f'touch {filename}')
        checkpoint = CheckpointInfo(filename, sha256)
        res = reload_model_weights(info=checkpoint)
        return res


def close_pil(image: Image):
    image.close()


def format_override_settings(override_settings):
    new_settings = []
    for item in override_settings or []:
        if 'sd_vae:' in item:
            item = str(item).replace("sd_vae", "VAE")
        new_settings.append(item)
    return new_settings


def save_processed_images(proc: Processed,
                          output_dir: str,
                          grid_dir: str,
                          script_dir: str,
                          task_id: str,
                          clean_upload_files: bool = True,
                          inspect: bool = False,
                          detect_multi_face: bool = False,
                          forbidden_review: bool = False):
    if not output_dir:
        raise ValueError('output is empty')

    date = datetime.today().strftime('%Y/%m/%d')
    output_dir = os.path.join(output_dir, date)
    grid_dir = os.path.join(grid_dir, date)
    script_dir = os.path.join(script_dir, date)

    out_grid_image = ImageOutput(OutImageType.Grid, grid_dir)
    out_image = ImageOutput(OutImageType.Image, output_dir)
    out_script_image = ImageOutput(OutImageType.Script, script_dir)
    faces = {}
    size = ''

    for n, processed_image in enumerate(proc.images):
        ex = '.png'

        if isinstance(processed_image, Image.Image) and hasattr(processed_image, 'already_saved_as'):
            saved_as = getattr(processed_image, 'already_saved_as')
            if saved_as:
                _, ex = os.path.splitext(saved_as)
        elif isinstance(processed_image, str):
            if not os.path.isfile(processed_image):
                raise OSError(f'cannot found processed image:{processed_image}')
            out_image.add_image(processed_image)
            continue

        if n < proc.index_of_first_image:
            filename = f"{task_id}{ex}"
            out_obj = out_grid_image
        elif n <= proc.index_of_end_image:
            filename = f"{task_id}-{n}{ex}"
            out_obj = out_image
        else:
            filename = f"{task_id}-{n}{ex}"
            out_obj = out_script_image

        if processed_image.mode == 'RGBA':
            processed_image = processed_image.convert("RGB")
        full_path = os.path.join(out_obj.output_dir, filename)

        pnginfo_data = PngInfo()
        pnginfo_data.add_text('by', 'xingzhe')
        size = f"{processed_image.width}*{processed_image.height}"
        infotexts = proc.infotexts[n].replace('-automatic1111', "-xingzhe") \
            if proc.infotexts and n < len(proc.infotexts) else ''
        for k, v in processed_image.info.items():
            if 'parameters' == k:
                v = str(v).replace('-automatic1111', "-xingzhe")
                print(f"image parameters:{v}")
                infotexts = v
                continue
            pnginfo_data.add_text(k, str(v))
        pnginfo_data.add_text('parameters', infotexts)

        processed_image.save(full_path, pnginfo=pnginfo_data)
        if detect_multi_face:
            # 忽略此图
            faces.update({
                os.path.basename(full_path): detect_faces(full_path)
            })

        out_obj.add_image(full_path)

    grid_keys = out_grid_image.multi_upload_keys(clean_upload_files)
    image_keys = out_image.multi_upload_keys(clean_upload_files)
    script_keys = out_script_image.multi_upload_keys(clean_upload_files)

    # if inspect:
    #     out_grid_image.inspect(grid_keys, forbidden_review)
    #     out_image.inspect(image_keys, forbidden_review)
    #
    output = {
        'grids': grid_keys.to_dict(),
        'samples': image_keys.to_dict()
    }
    #
    # all_keys = grid_keys + image_keys + script_keys

    forbidden_keys = {}
    if inspect:
        forbidden_keys = out_image.inspect(image_keys, forbidden_review)

    all_keys = grid_keys + image_keys + script_keys
    has_grid = proc.index_of_first_image > 0
    output.update({
        'has_grid': has_grid,
        'all': all_keys.to_dict(forbidden_keys, has_grid),
        'size': size,
        'face': faces
    })

    return output


# 针对衫数人物
def save_processed_images_shanshu(proc: Processed, output_dir: str, grid_dir: str, script_dir: str,
                                  task_id: str, clean_upload_files: bool = True, inspect=False):
    if not output_dir:
        raise ValueError('output is empty')

    date = datetime.today().strftime('%Y/%m/%d')
    output_dir = os.path.join(output_dir, date)
    grid_dir = os.path.join(grid_dir, date)
    script_dir = os.path.join(script_dir, date)

    out_grid_image = ImageOutput(OutImageType.Grid, grid_dir)
    out_image = ImageOutput(OutImageType.Image, output_dir)
    out_script_image = ImageOutput(OutImageType.Script, script_dir)

    size = ''
    for n, processed_image in enumerate(proc.images):
        ex = '.png'
        if isinstance(processed_image, Image.Image) and hasattr(processed_image, 'already_saved_as'):
            saved_as = getattr(processed_image, 'already_saved_as')
            if saved_as:
                _, ex = os.path.splitext(saved_as)

        if n < proc.index_of_first_image:
            filename = f"{task_id}{ex}"
            out_obj = out_grid_image
        elif n <= proc.index_of_end_image:
            filename = f"{task_id}-{n}{ex}"
            out_obj = out_image
        else:
            filename = f"{task_id}-{n}{ex}"
            out_obj = out_script_image

        # if processed_image.mode == 'RGBA':
        #     processed_image = processed_image.convert("RGB")
        full_path = os.path.join(out_obj.output_dir, filename)

        pnginfo_data = PngInfo()
        pnginfo_data.add_text('by', 'xingzhe')
        size = f"{processed_image.width}*{processed_image.height}"
        infotexts = proc.infotexts[n].replace('-automatic1111', "-xingzhe") \
            if proc.infotexts and n < len(proc.infotexts) else ''
        for k, v in processed_image.info.items():
            if 'parameters' == k:
                v = str(v).replace('-automatic1111', "-xingzhe")
                print(f"image parameters:{v}")
                infotexts = v
                continue
            pnginfo_data.add_text(k, str(v))
        pnginfo_data.add_text('parameters', infotexts)

        processed_image.save(full_path, pnginfo=pnginfo_data)
        out_obj.add_image(full_path)

    grid_keys = out_grid_image.multi_upload_keys(clean_upload_files)
    image_keys = out_image.multi_upload_keys(clean_upload_files)
    script_keys = out_script_image.multi_upload_keys(clean_upload_files)

    if inspect:
        out_grid_image.inspect(grid_keys)
        out_image.inspect(image_keys)

    output = {
        'grids': grid_keys.to_dict(),
        'samples': image_keys.to_dict()
    }

    all_keys = grid_keys + image_keys + script_keys
    output.update({
        'has_grid': proc.index_of_first_image > 0,
        'all': all_keys.to_dict(),
        'size': size
    })

    return output
