#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 9:31 AM
# @Author  : wangdongming
# @Site    : 
# @File    : doppelganger.py
# @Software: Hifive
import math
import typing
import shutil
import os.path
import random
import numpy as np
from loguru import logger
from modules.shared import mem_mon as vram_mon
from trainx.utils import Tmp, detect_image_face, upload_files
from trainx.typex import DigitalDoppelgangerTask, Task, TrainLoraTask, PreprocessTask
from sd_scripts.train_auto_xz import train_auto
from worker.task import Task, TaskStatus, TaskProgress, TrainEpoch
from trainx.utils import calculate_sha256
from modules.devices import torch_gc
# class DigitalDoppelganger:


def digital_doppelganger(job: Task, dump_func: typing.Callable = None):
    p = TaskProgress.new_prepare(job, 'prepare')
    p.eta_relative = len(job)
    yield p

    task = DigitalDoppelgangerTask(job)
    eta = int(len(task.image_keys) * 0.89 + 1416)
    p = TaskProgress.new_ready(job, 'ready preprocess', 0)
    p.eta_relative = eta
    yield p

    logger.debug(">> download images...")
    target_dir = os.path.join(Tmp, job.id)
    os.makedirs(target_dir, exist_ok=True)

    image_dir = task.download_move_input_images()
    logger.debug(f">> input images dir:{image_dir}")

    if image_dir:
        # 检测年龄
        images = [os.path.join(image_dir, os.path.basename(i)) for i in random.choices(task.image_keys, k=5)]
        face_info = []
        for n, face in detect_image_face(*images):
            if face:
                face_info.append((face.gender, face.age))

        age = np.average([x[-1] for x in face_info])
        gender = '1girl' if np.sum([x[0] == 0 for x in face_info]) > len(face_info) / 2 else '1boy'

        p = TaskProgress.new_running(job, 'train running.', 1)
        p.eta_relative = eta
        yield p

        def train_progress_callback(epoch, loss, num_train_epochs, progress):
            progress = progress if progress > 1 else progress * 100
            if progress - p.task_progress >= 5:

                free, total = vram_mon.cuda_mem_get_info()
                logger.info(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')

                p.task_progress = min(progress, 97)

                previous_eta = p.eta_relative

                p.calc_eta_relative()

                # 防止ETA比上一轮数字大~
                logger.debug(f"previous eta:{previous_eta}, calculate eta:{p.eta_relative}")
                if previous_eta < p.eta_relative:
                    epoch = progress // 10  # 控制的是10epoch
                    eta_relative = eta - epoch * 60
                    p.eta_relative = min(eta_relative, previous_eta - 60)
                    logger.debug(f"===>>> new eta:{eta_relative}")

                #  time_since_start = time.time() - shared.state.time_start
                #         eta = (time_since_start / progress)
                #         eta_relative = eta - time_since_start
                # logger.info(f"eta: {p.eta_relative}S ({eta} - {time_since_start}) ")
                if callable(dump_func):
                    dump_func(p)

        logger.debug(f">> preprocess and train....")
        out_path, gender_tag = train_auto(
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
                'models': [],
                'gender': gender,
                'age': int(age) if age and not math.isnan(age) else 20
            }

            cover = task.get_model_cover_key()
            dirname = os.path.dirname(out_path)
            basename = os.path.basename(out_path)
            without, ex = os.path.splitext(basename)

            sha256 = calculate_sha256(out_path, 1024 * 1024 * 512)
            hash_file_path = os.path.join(dirname, sha256 + ex)

            shutil.move(out_path, hash_file_path)
            key = upload_files(False, hash_file_path, dirname='models/digital/Lora', task_id=task.id)
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
        print(f"remove image dir:{image_dir}")
        shutil.rmtree(image_dir)
    else:
        p = TaskProgress.new_failed(job, 'train failed(cannot download images)')
        yield p
