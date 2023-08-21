#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 3:53 PM
# @Author  : wangdongming
# @Site    : 
# @File    : gen_fusion_img.py
# @Software: Hifive
import time
import typing
from modules import shared
from enum import IntEnum
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path

GenRefineImageMatCount = 1  # 垫图数


class RefineTaskType(IntEnum):
    GenRefineImage = 0


class GenRefineImageTask(Txt2ImgTask):

    def __init__(self, *args, **kwargs):
        super(GenRefineImageTask, self).__init__(*args, **kwargs)
        # 文生图默认生成一张~
        self.n_iter = GenRefineImageMatCount
        self.batch_size = 1


class RefineTaskHandler(Txt2ImgTaskHandler):

    def __init__(self):
        super(RefineTaskHandler, self).__init__()
        self.task_type = TaskType.RefineImage

    def _build_gen_refine_arg(self, progress: TaskProgress) -> GenRefineImageTask:
        self._refresh_default_script_args()
        t = GenRefineImageTask.from_task(progress.task, self.default_script_args)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 0)
        return t

    def _gen_refine_cb(self, progress: TaskProgress, index: int = 0):
        if shared.state.sampling_step - shared.state.current_image_sampling_step < 5:
            return
        p = 0

        if shared.state.job_count > 0:
            job_no = shared.state.job_no - 1 if shared.state.job_no > 0 else 0
            p += job_no / (progress.task['n_iter'] * progress.task['batch_size'] + GenRefineImageMatCount)
            # p += (shared.state.job_no) / shared.state.job_count
        if shared.state.sampling_steps > 0:
            p += 1 / (progress.task['n_iter'] * progress.task[
                'batch_size']) * shared.state.sampling_step / shared.state.sampling_steps

        off = index * 20
        ratio = max(0.2 * (index + 1), 0.98)
        current_progress = int((off + p) * ratio)
        if current_progress < progress.task_progress:
            return

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start / p)
        progress.task_progress = current_progress
        progress.eta_relative = int(eta - time_since_start)

        self._set_task_status(progress)

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        pass

    def _exec_refine_image(self, task: Task) -> typing.Iterable[TaskProgress]:
        base_model_path = self._get_local_checkpoint(task)
        load_sd_model_weights(base_model_path, task.model_hash)
        progress = TaskProgress.new_ready(task, f'model loaded, gen refine image...')
        yield progress
        process_args = self._build_gen_refine_arg(progress)
        self._set_little_models(process_args)
        progress.status = TaskStatus.Running
        progress.task_desc = f't2i task({task.id}) running'
        yield progress
        shared.state.begin()
        # shared.state.job_count = process_args.n_iter * process_args.batch_size
        # 生成一张图
        processed = process_images(process_args)

        # i2i
        processed_i2i_1 = StableDiffusionProcessingImg2Img(init_images=processed.images, denoising_strength=0.15)
        processed_i2i_2 = StableDiffusionProcessingImg2Img(init_images=processed.images, denoising_strength=0.25)
        processed_i2i_3 = StableDiffusionProcessingImg2Img(init_images=processed.images, denoising_strength=0.35)
        processed_i2i_4 = StableDiffusionProcessingImg2Img(init_images=processed.images, denoising_strength=0.45)

        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 1)
        processed_1 = process_images(processed_i2i_1)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 2)
        processed_2 = process_images(processed_i2i_2)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 3)
        processed_3 = process_images(processed_i2i_3)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 4)
        processed_4 = process_images(processed_i2i_4)

        images = list(processed_1.images)
        images.extend(processed_2.images)
        images.extend(processed_3.images)
        images.extend(processed_4.images)

        shared.state.end()
        process_args.close()

        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed_4,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       task.id,
                                       inspect=process_args.kwargs.get("need_audit", False))

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed.all_seeds, processed.all_subseeds)

        yield progress
