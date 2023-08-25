#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/25 2:43 PM
# @Author  : wangdongming
# @Site    : 
# @File    : digital.py
# @Software: Hifive
import random
import time
import typing
import modules
from modules import shared
from enum import IntEnum
from PIL import ImageOps
from handlers.img2img import Img2ImgTask, Img2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path


class DigitalTaskType(IntEnum):
    Img2Img = 1


class DigitalTaskHandler(Img2ImgTaskHandler):

    def __init__(self):
        super(DigitalTaskHandler, self).__init__()
        self.task_type = TaskType.Digital

    def _build_i2i_tasks(self, t: Task):
        tasks = []
        for denoising_strength in [0.15, 0.25, 0.35, 0.45]:
            t['denoising_strength'] = denoising_strength
            t['n_iter'] = 1
            t['batch_size'] = 1
            tasks.append(Img2ImgTask.from_task(t, self.default_script_args))

        return tasks

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        if task.minor_type == DigitalTaskType.Img2Img:
            yield from self._exec_img2img(task)

    def _exec_img2img(self, task: Task) -> typing.Iterable[TaskProgress]:
        base_model_path = self._get_local_checkpoint(task)
        load_sd_model_weights(base_model_path, task.model_hash)
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...')
        self._refresh_default_script_args()
        yield progress
        tasks = self._build_i2i_tasks(task)
        # i2i
        images = []
        all_seeds = []
        all_subseeds = []
        processed = None
        for i, p in enumerate(tasks):
            if i == 0:
                self._set_little_models(p)
            processed = process_images(p)
            all_seeds.extend(processed.all_seeds)
            all_subseeds.extend(processed.all_subseeds)
            images.append(processed.images[0])
            progress.task_progress = min((i + 1) * 100 / len(tasks), 98)
            yield progress
            p.close()

        grid = modules.images.image_grid(images, len(images))
        images.insert(0, grid)
        processed.index_of_first_image = 1
        processed.index_of_end_image = len(images)
        processed.images = images

        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed,
                                       tasks[0].outpath_samples,
                                       tasks[0].outpath_grids,
                                       tasks[0].outpath_scripts,
                                       task.id,
                                       inspect=False)

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(all_seeds, all_subseeds)

        yield progress
