#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 9:31 AM
# @Author  : wangdongming
# @Site    : 
# @File    : doppelganger.py
# @Software: Hifive
import copy
import os.path
import shutil
import typing

import math
from PIL import Image
from loguru import logger
from worker.task import Task, TaskStatus, TaskProgress, TrainEpoch
from trainx.utils import get_tmp_local_path, Tmp, upload_files
from tools.file import zip_compress, zip_uncompress, find_files_from_dir
from trainx.typex import DigitalDoppelgangerTask, Task, TrainLoraTask, PreprocessTask
from modules.textual_inversion.preprocess import preprocess_sub_dir
from sd_scripts.train_network_all import train_with_params
from trainx.lora import *


# class DigitalDoppelganger:


def download_images(task: PreprocessTask):
    local_files = []
    for i, image in enumerate(task.image_keys):
        if len(local_files) > 60:
            continue
        target_dir = os.path.join(Tmp, task.id)
        file = get_tmp_local_path(image, dir=target_dir)
        if os.path.isfile(file):
            local_files.append(file)
    return local_files


def digital_doppelganger(job: Task, dump_func: typing.Callable = None):
    task = PreprocessTask(job)
    task['process_width'] = 512
    task['process_height'] = 768
    p = TaskProgress.new_ready(job, 'ready preprocess')
    yield p

    logger.debug(">> download images...")
    target_dir = os.path.join(Tmp, job.id)
    os.makedirs(target_dir, exist_ok=True)
    images = download_images(task)

    if images:

        processed_dir = os.path.join(Tmp, job.id + '-preprocessed')
        params = copy.copy(task.params)

        def progress_callback(progress):
            p = TaskProgress.new_running(job, 'run preprocess', int(progress * 0.2))
            if callable(dump_func):
                dump_func(p)

        params.update(
            {
                'progress_cb': progress_callback
            }
        )
        logger.debug(">>> preprocess images...")
        if not task.ignore:
            preprocess_sub_dir(
                target_dir,
                processed_dir,
                **params
            )
        else:
            processed_dir = target_dir

        p = TaskProgress.new_running(job, 'train ready', 30)
        yield p
        # resolution, min_reso, max_reso = get_batch_image_size(processed_dir)
        train_lora_task = TrainLoraTask(job)
        kwargs = train_lora_task.build_command_args()

        logger.info("=============>>>> start train lora <<<<=============")
        logger.info(">>> command args:")
        for k, v in kwargs.items():
            logger.info(f"> args: {k}: {v}")
        logger.info("====================================================")
        p = TaskProgress.new_running(job, 'running', 0)

        def train_progress_callback(epoch, loss, num_train_epochs):
            print(f">>> update progress, epoch:{epoch},loss:{loss},len:{len(p.train.epoch)}")
            progress = 30 + epoch / num_train_epochs * 100 * 0.6
            p.train.add_epoch_log(TrainEpoch(epoch, loss))
            p.task_progress = progress
            if math.isnan(loss) or str(loss).lower() == 'nan':
                p.status = TaskStatus.Failed
            if callable(dump_func):
                dump_func(p)

        ok = train_with_params(callback=train_progress_callback, **kwargs)
        # start_train_process(task, kwargs, dump_func)
        # local_models = get_train_models(train_lora_task, kwargs['output_name'])
        # ok = local_models is not None and len(local_models) > 0
        torch_gc()
        local_models = get_train_models(train_lora_task, kwargs['output_name'])
        if ok:
            logger.info("=============>>>> end of train <<<<=============")
            # material = train_lora_task.compress_train_material(p.train.format_epoch_log(), kwargs)
            result = {
                'material': None,
                'models': []
            }

            cover = train_lora_task.get_model_cover_key()
            for m in local_models:
                # rename
                dirname = os.path.dirname(m)
                basename = os.path.basename(m)
                without, ex = os.path.splitext(basename)
                # sha256 = SHA256.new(basename.encode()).hexdigest()
                sha256 = calculate_sha256(m, 1024 * 1024 * 512)
                array = without.split('-')
                epoch = array[-1] if len(array) > 1 else ''
                hash_file_path = os.path.join(dirname, sha256 + ex)

                shutil.move(m, hash_file_path)
                key = upload_files(False, hash_file_path)
                result['models'].append({
                    'key': key[0] if key else '',
                    'thumbnail_path': cover,
                    'hash': sha256,
                    'epoch': epoch
                })

            # if os.path.isfile(material):
            #     material_keys = upload_files(False, material)
            #     result['material'] = material_keys[0] if material_keys else ''
            # # notify web server
            # sender = RedisSender()
            # sender.notify_train_task(job)

            fp = TaskProgress.new_finish(job, {
                'train': result
            }, False)
            fp.train = p.train

            yield fp
        else:
            p = TaskProgress.new_failed(job, 'train failed(unknown errors)')
            yield p
