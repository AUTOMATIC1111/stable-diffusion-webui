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
import time
import typing

from modules.shared import mem_mon as vram_mon
from trainx.utils import get_tmp_local_path, Tmp, upload_files
from tools.file import zip_compress, zip_uncompress, find_files_from_dir
from trainx.typex import DigitalDoppelgangerTask, Task, TrainLoraTask, PreprocessTask
from sd_scripts.train_auto_xz import train_auto
from trainx.lora import *


# class DigitalDoppelganger:


def digital_doppelganger(job: Task, dump_func: typing.Callable = None):
    task = DigitalDoppelgangerTask(job)
    p = TaskProgress.new_ready(job, 'ready preprocess')
    p.eta_relative = 40 * 60
    yield p

    logger.debug(">> download images...")
    target_dir = os.path.join(Tmp, job.id)
    os.makedirs(target_dir, exist_ok=True)

    image_dir = task.download_move_input_images()
    logger.debug(f">> input images dir:{image_dir}")

    if image_dir:
        p = TaskProgress.new_running(job, 'train running.')
        p.eta_relative = 35 * 60
        yield p
        time_start = time.time()

        def train_progress_callback(progress):
            progress = progress if progress > 1 else progress * 100
            if progress - p.task_progress >= 2:
                time_since_start = time.time() - time_start
                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                p.task_progress = min(progress, 97)
                eta = (time_since_start*100 / p.task_progress)
                p.eta_relative = int(eta - time_since_start)
                #  time_since_start = time.time() - shared.state.time_start
                #         eta = (time_since_start / progress)
                #         eta_relative = eta - time_since_start
                logger.info(f"eta: {p.eta_relative}S ({eta} - {time_since_start}) ")
                if callable(dump_func):
                    dump_func(p)

        logger.debug(f">> preprocess and train....")
        out_path = train_auto(
            train_callback=train_progress_callback,
            train_data_dir=image_dir,
            train_type=task.train_type,
            task_id=task.id,
            sd_model_path=task.base_model,
            lora_path=task.output_dir,
            general_model_path=task.general_model_path,
        )

        torch_gc()
        logger.debug(f">> train complete: {out_path}")
        if out_path and os.path.isfile(out_path):
            result = {
                'material': None,
                'models': []
            }

            cover = task.get_model_cover_key()
            dirname = os.path.dirname(out_path)
            basename = os.path.basename(out_path)
            without, ex = os.path.splitext(basename)

            sha256 = calculate_sha256(out_path, 1024 * 1024 * 512)
            hash_file_path = os.path.join(dirname, sha256 + ex)

            shutil.move(out_path, hash_file_path)
            key = upload_files(False, hash_file_path)
            result['models'].append({
                'key': key[0] if key else '',
                'thumbnail_path': cover,
                'hash': sha256,
            })

            fp = TaskProgress.new_finish(job, {
                'train': result
            }, False)
            fp.train = p.train

            yield fp
        else:
            p = TaskProgress.new_failed(job, 'train failed(unknown errors)')
            yield p
    else:
        p = TaskProgress.new_failed(job, 'train failed(cannot download images)')
        yield p

