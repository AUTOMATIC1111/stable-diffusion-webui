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


def direct_train(task: TrainLoraTask):
    def progress_callback(epoch, loss):
        print(f"> task:{task.id}, epoch:{epoch}, loss:{loss}")

    args = task.build_command_args()
    parser = setup_parser()
    parser.parse_args(args)
    train(args, progress_callback)


def get_train_models(train_lora_task: TrainLoraTask):
    models = []
    for file in os.listdir(train_lora_task.output_dir):
        if os.path.isfile(file):
            _, ex = os.path.splitext(file)
            if ex.lower() == '.safetensors':
                models.append(file)

    def sort_model(module_path):
        basename, _ = os.path.splitext(os.path.basename(module_path))
        if '-' not in basename:
            return -1
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
    args, kwargs = train_lora_task.build_command_args()
    p = TaskProgress.new_ready(task, 'ready')
    yield p
    logger.info("=============>>>> start train lora <<<<=============")
    logger.info(">>> command args:")
    for arg in args:
        logger.info(arg)

    train_with_params(callback=callback, **kwargs)
    material = train_lora_task.compress_train_material()
    result = {
        'material': None,
        'models': []
    }

    if os.path.isfile(material):
        result['material'] = upload_files(False, material)

    local_models = get_train_models(train_lora_task)
    for m in local_models:
        key = upload_files(False, m)
        result['models'].append(key)

    p = TaskProgress.new_finish(task, {
        'train': result
    })

    yield p
