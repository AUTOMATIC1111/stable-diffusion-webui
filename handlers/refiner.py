#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 3:53 PM
# @Author  : wangdongming
# @Site    : 
# @File    : gen_fusion_img.py
# @Software: Hifive
import random
import time
import typing
import modules
from modules import shared
from enum import IntEnum
from PIL import ImageOps
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path

GenRefineImageMatCount = 1  # 垫图数


class RefineTaskType(IntEnum):
    GenRefineImage = 1


class GenRefineImageTask(Txt2ImgTask):

    def __init__(self, *args, **kwargs):
        super(GenRefineImageTask, self).__init__(*args, **kwargs)
        # 文生图默认生成一张~
        # self.n_iter = GenRefineImageMatCount
        # self.batch_size = 1


class RefineTaskHandler(Txt2ImgTaskHandler):

    def __init__(self):
        super(RefineTaskHandler, self).__init__()
        self.task_type = TaskType.RefineImage

    def _build_gen_refine_arg(self, progress: TaskProgress) -> GenRefineImageTask:
        self._refresh_default_script_args()
        t = GenRefineImageTask.from_task(progress.task, self.default_script_args)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 0)
        return t

    def _build_gen_refine_i2i_args(self, t: GenRefineImageTask, processed: Processed):
        denoising_strength = random.choice((0.15, 0.25, 0.3, 0.35, 0.4))
        return StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=t.outpath_samples,
            outpath_grids=t.outpath_grids,
            outpath_scripts=t.outpath_grids,
            prompt=t.prompt,
            negative_prompt=t.negative_prompt,
            seed=-1,
            subseed=-1,
            sampler_name=t.sampler_name,
            batch_size=1,
            n_iter=4,
            steps=t.steps,
            cfg_scale=3,
            width=t.width,
            height=t.height,
            restore_faces=t.restore_faces,
            tiling=t.tiling,
            init_images=processed.images,
            mask=None,
            denoising_strength=denoising_strength
        )

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

        off = index * 12.5
        ratio = min(0.125 * (index + 1), 0.98)
        current_progress = int((off + p*100) * ratio)
        if current_progress < progress.task_progress:
            return

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start / p)
        progress.task_progress = current_progress
        progress.eta_relative = int(eta - time_since_start)

        self._set_task_status(progress)

    def batch_process_i2i(self, p):
        fix_seed(p)

        images = p.init_images

        save_normally = True

        p.do_not_save_grid = True
        p.do_not_save_samples = not save_normally

        shared.state.job_count = len(images) * p.n_iter

        outs = []
        for i, image in enumerate(images):
            shared.state.job = f"{i + 1} out of {len(images)}"
            if shared.state.skipped:
                shared.state.skipped = False

            if shared.state.interrupted:
                break

            # Use the EXIF orientation of photos taken by smartphones.
            img = ImageOps.exif_transpose(image)
            p.init_images = [img] * p.batch_size

            proc = modules.scripts.scripts_img2img.run(p)
            if proc is None:
                proc = process_images(p)

            outs.extend(proc.images)
        return outs

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        if task.minor_type == RefineTaskType.GenRefineImage:
            yield from self._exec_refine_image(task)

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
        # 生成垫图
        processed = process_images(process_args)

        # i2i
        processed_i2i_1 = self._build_gen_refine_i2i_args(process_args, processed)
        shared.state.current_latent_changed_callback = lambda: self._gen_refine_cb(progress, 4)

        processed_i2i = self.batch_process_i2i(processed_i2i_1)

        shared.state.end()
        process_args.close()

        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed_i2i,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       task.id,
                                       inspect=process_args.kwargs.get("need_audit", False))

        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed_i2i.all_seeds, processed_i2i.all_subseeds)

        yield progress
