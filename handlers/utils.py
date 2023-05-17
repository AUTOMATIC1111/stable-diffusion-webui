#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 2:51 PM
# @Author  : wangdongming
# @Site    : 
# @File    : utils.py
# @Software: Hifive
import os
import typing
from PIL import Image
from loguru import logger
from datetime import datetime
from worker.task_recv import Tmp
from PIL.PngImagePlugin import PngInfo
from tools.encryptor import des_encrypt
from modules.processing import Processed
from modules.scripts import Script, ScriptRunner
from modules.sd_models import reload_model_weights, CheckpointInfo
from handlers.formatter import format_alwayson_script_args
from tools.environment import get_file_storage_system_env, Env_BucketKey, S3ImageBucket, S3Tmp, S3SDWEB
from filestorage import FileStorageCls, get_local_path, batch_download
from handlers.typex import ModelLocation, ModelType, ImageOutput, OutImageType, UserModelLocation


StrMapMap = typing.Mapping[str, typing.Mapping[str, typing.Any]]


def clean_models():
    pass


def get_model_local_path(remoting_path: str, model_type: ModelType):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path
    # 判断user-models下
    os.makedirs(UserModelLocation[model_type], exist_ok=True)
    dst = os.path.join(UserModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        return dst

    os.makedirs(ModelLocation[model_type], exist_ok=True)
    dst = os.path.join(ModelLocation[model_type], os.path.basename(remoting_path))
    if os.path.isfile(dst):
        return dst

    dst = get_local_path(remoting_path, dst)
    if os.path.isfile(dst):
        if model_type == ModelType.CheckPoint:
            checkpoint = CheckpointInfo(dst)
            checkpoint.register()
        return dst


def batch_model_local_paths(model_type: ModelType, *remoting_paths: str) \
        -> typing.Sequence[str]:
    remoting_key_dst_pairs, loc = [], []
    os.makedirs(ModelLocation[model_type], exist_ok=True)
    for p in remoting_paths:
        if os.path.isfile(p):
            loc.append(p)
            continue
        dst = os.path.join(ModelLocation[model_type], os.path.basename(p))
        loc.append(dst)
        if not os.path.isfile(dst):
            remoting_key_dst_pairs.append((p, dst))
    batch_download(remoting_key_dst_pairs)
    return loc


def get_tmp_local_path(remoting_path: str):
    if not remoting_path:
        raise OSError(f'remoting path is empty')
    if os.path.isfile(remoting_path):
        return remoting_path

    os.makedirs(Tmp, exist_ok=True)
    dst = os.path.join(Tmp, os.path.basename(remoting_path))
    return get_local_path(remoting_path, dst)


def upload_files(is_tmp, *files):
    keys = []
    if files:
        date = datetime.today().strftime('%Y/%m/%d')
        storage_env = get_file_storage_system_env()
        bucket = storage_env.get(Env_BucketKey) or S3ImageBucket
        file_storage_system = FileStorageCls()
        relative = S3Tmp if is_tmp else S3SDWEB

        for f in files:
            name = os.path.basename(f)
            key = os.path.join(bucket, relative, date, name)
            file_storage_system.upload(f, key)
            keys.append(key)
    return keys


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
        return [script.title().lower().replace(' ', '-') for script in scripts].index(name.lower())
    except:
        raise Exception(f"Script '{name}' not found")


def init_script_args(default_script_args: typing.Sequence, alwayson_scripts: StrMapMap, selectable_scripts: Script,
                     selectable_idx: int, request_script_args: typing.Sequence, script_runner: ScriptRunner):
    script_args = [x for x in default_script_args]

    if selectable_scripts:
        script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request_script_args
        script_args[0] = selectable_idx + 1

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
            if "args" in alwayson_scripts[alwayson_script_name]:
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


def load_sd_model_weights(filename, sha256=None):
    checkpoint = CheckpointInfo(filename, sha256)
    return reload_model_weights(info=checkpoint)


def save_processed_images(proc: Processed, output_dir: str, grid_dir: str,
                          script_dir: str, task_id: str, clean_upload_files: bool = True):
    if not output_dir:
        raise ValueError('output is empty')

    date = datetime.today().strftime('%Y/%m/%d')
    output_dir = os.path.join(output_dir, date)
    grid_dir = os.path.join(grid_dir, date)
    script_dir = os.path.join(script_dir, date)

    out_grid_image = ImageOutput(OutImageType.Grid, grid_dir)
    out_image = ImageOutput(OutImageType.Image, output_dir)
    out_script_image = ImageOutput(OutImageType.Script, script_dir)

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

        if processed_image.mode == 'RGBA':
            processed_image = processed_image.convert("RGB")
        full_path = os.path.join(out_obj.output_dir, filename)

        pnginfo_data = PngInfo()
        pnginfo_data.add_text('by', 'xing-zhe')
        for k, v in processed_image.info.items():
            if 'parameters' == k:
                v = des_encrypt(v)
            pnginfo_data.add_text(k, str(v))

        processed_image.save(full_path, pnginfo=pnginfo_data)
        out_obj.add_image(full_path)

    grid_keys = out_grid_image.upload_keys(clean_upload_files)
    image_keys = out_image.upload_keys(clean_upload_files)
    script_keys = out_script_image.upload_keys(clean_upload_files)

    output = {
        'grids': grid_keys,
        'samples': image_keys
    }

    all_keys = grid_keys + image_keys + script_keys
    output.update({
        'has_grid': proc.index_of_first_image > 0,
        'all': all_keys,
    })

    return output
