#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 9:06 PM
# @Author  : wangdongming
# @Site    : 
# @File    : train_lora.py
# @Software: Hifive
import os.path
import typing
from loguru import logger
from worker.task import Task, TaskProgress
from sd_scripts.train_network_ly import train_with_params
from .typex import TrainLoraTask
from .utils import upload_files


def get_train_models(train_lora_task: TrainLoraTask, model_name: str):
    models = []
    for file in os.listdir(train_lora_task.output_dir):
        full = os.path.join(train_lora_task.output_dir, file)
        if os.path.isfile(full) and file.startswith(model_name):
            _, ex = os.path.splitext(file)
            if ex.lower() == '.safetensors':
                models.append(full)

    def sort_model(module_path):
        basename, _ = os.path.splitext(os.path.basename(module_path))

        if '-' not in basename:
            return 10000001
        seg = basename.split('-')[-1]
        if seg == 'last':
            return 10000000
        try:
            x = int(seg)
        except:
            x = 0
        return x

    return sorted(models, key=sort_model)


def exec_train_lora_task(task: Task, callback: typing.Callable = None):
    train_lora_task = TrainLoraTask(task)
    kwargs = train_lora_task.build_command_args()
    p = TaskProgress.new_ready(task, 'ready')
    yield p
    logger.info("=============>>>> start train lora <<<<=============")
    logger.info(">>> command args:")
    for k, v in kwargs.items():
        logger.info(f"> args: {k}: {v}")

    train_with_params(callback=callback, **kwargs)
    material = train_lora_task.compress_train_material()
    result = {
        'material': None,
        'models': []
    }

    if os.path.isfile(material):
        result['material'] = upload_files(False, material)

    local_models = get_train_models(train_lora_task, kwargs['output_name'])
    cover = train_lora_task.get_model_cover_key()
    for m in local_models:
        key = upload_files(False, m)
        result['models'].append({
            'key': key,
            'thumbnail_path': cover,
            'hash': train_lora_task.hash_id
        })

    p = TaskProgress.new_finish(task, {
        'train': result
    }, True)

    yield p
