#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 2:51 PM
# @Author  : wangdongming
# @Site    : 
# @File    : utils.py
# @Software: Hifive
import os
import socket
import typing
from PIL import Image
from loguru import logger
from datetime import datetime
from modules.scripts import Script
from modules.sd_models import reload_model_weights, CheckpointInfo
from handlers.formatter import format_alwayson_script_args

StrMapMap = typing.Mapping[str, typing.Mapping[str, typing.Any]]


def get_host_ip():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        if s:
            s.close()
    return ip


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
        return [script.title().lower() for script in scripts].index(name.lower())
    except:
        raise Exception(f"Script '{name}' not found")


def init_script_args(default_script_args: typing.Sequence, alwayson_scripts: StrMapMap, selectable_scripts: Script,
                     selectable_idx, request_script_args, script_runner):
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


def load_sd_model_weights(filename):
    checkpoint = CheckpointInfo(filename)
    return reload_model_weights(info=checkpoint)


def save_processed_images(proc, output_dir, task_id):
    save_normally = output_dir == ''
    local_images = []
    date = datetime.today().strftime('%Y-%m-%d')
    output_dir = os.path.join(output_dir, date)

    for n, processed_image in enumerate(proc.images):
        ex = '.png'
        if isinstance(processed_image, Image.Image) and hasattr(processed_image, 'already_saved_as'):
            saved_as = getattr(processed_image, 'already_saved_as')
            if saved_as:
                _, ex = os.path.splitext(saved_as)

        if n > 0:
            filename = f"{task_id}-{n}{ex}"
        else:
            filename = f"{task_id}{ex}"
        if not save_normally:
            os.makedirs(output_dir, exist_ok=True)
            if processed_image.mode == 'RGBA':
                processed_image = processed_image.convert("RGB")
            full_path = os.path.join(output_dir, filename)
            processed_image.save(full_path)
            local_images.append(full_path)
    return local_images
