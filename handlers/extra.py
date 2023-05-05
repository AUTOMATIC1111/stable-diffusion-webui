#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 8:45 PM
# @Author  : wangdongming
# @Site    : 
# @File    : extra.py
# @Software: Hifive
import os.path
import typing
from PIL import Image
from enum import IntEnum
from handlers.dumper import dumper
from modules.postprocessing import run_extras
from handlers.utils import get_tmp_local_path, Tmp, upload_tmp_files
from worker.task import Task, TaskType, TaskHandler, TaskProgress, TaskStatus


class ExtraTaskMinorTaskType(IntEnum):
    Default = 0
    SingleUpscaler = 1


class SingleUpscalerTask:

    def __init__(self,
                 image: str,  # 图片路径
                 gfpgan_visibility: float = 0,  # gfpgan 可见度，0-1
                 codeformer_visibility: float = 0,  # code former 可见度，0-1
                 codeformer_weight: float = 0,  # code former 权重
                 upscaling_resize: float = 2,  # 等比缩放,1-8
                 upscaling_resize_w: int = 512,  # 等比缩放 w
                 upscaling_resize_h: int = 512,  # 等比缩放 h
                 upscaling_crop: bool = True,  # 裁剪以适应宽高比
                 extras_upscaler_1: str = None,  # upscaler_1
                 extras_upscaler_2: str = None,  # upscaler_2
                 extras_upscaler_2_visibility: float = 0  # upscaler_2 可见度，0-1
                 ):
        self.resize_mode = 0  # 模式 0-single
        self.extras_mode = 0
        self.image_folder = ""
        self.input_dir = ""
        self.output_dir = ""
        self.save_output = False
        self.show_extras_results = True
        self.gfpgan_visibility = gfpgan_visibility
        self.codeformer_visibility = codeformer_visibility
        self.codeformer_weight = codeformer_weight
        self.upscaling_resize = upscaling_resize
        self.upscaling_resize_w = upscaling_resize_w
        self.upscaling_resize_h = upscaling_resize_h
        self.upscaling_crop = upscaling_crop
        self.extras_upscaler_1 = extras_upscaler_1
        self.extras_upscaler_2 = extras_upscaler_2
        self.extras_upscaler_2_visibility = extras_upscaler_2_visibility

        local_image = get_tmp_local_path(image)
        if local_image and os.path.isfile(local_image):
            self.image = Image.open(local_image)
        else:
            raise OSError(f'cannot found image:{image}')

    @classmethod
    def exec_task(cls, task: Task):
        t = SingleUpscalerTask(
            task['image'],
            task.get('gfpgan_visibility', 0),
            task.get('codeformer_visibility', 0),
            task.get('codeformer_weight', 0),
            task.get('upscaling_resize', 2),
            task.get('upscaling_resize_w', 512),
            task.get('upscaling_resize_h', 512),
            task.get('upscaling_crop', True),
            task.get('extras_upscaler_1'),
            task.get('extras_upscaler_2'),
            task.get('extras_upscaler_2_visibility', 0)
        )
        return run_extras(t.extras_mode, t.resize_mode, t.image, t.image_folder, t.input_dir, t.output_dir,
                          t.show_extras_results, t.gfpgan_visibility, t.codeformer_visibility,
                          t.codeformer_weight, t.upscaling_resize, t.upscaling_resize_w,
                          t.upscaling_resize_h, t.upscaling_crop, t.extras_upscaler_1,
                          t.extras_upscaler_2, t.extras_upscaler_2_visibility, False)


class ExtraTaskHandler(TaskHandler):

    def __init__(self):
        super(ExtraTaskHandler, self).__init__(TaskType.Extra)

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        minor_type = ExtraTaskMinorTaskType(task.minor_type)
        if minor_type <= ExtraTaskMinorTaskType.SingleUpscaler:
            yield from self.__exec_single_upscaler_task(task)

    def __exec_single_upscaler_task(self, task: Task):
        p = TaskProgress.new_ready(task, f"ready exec upscaler, task:{task.id}")
        yield p
        result = SingleUpscalerTask.exec_task(task)
        if result:
            images = self._save_images(task, result)
            keys = upload_tmp_files(*images)
            p.set_finish_result(keys)
            p.task_desc = f'upscaler task:{task.id} finished.'
        else:
            p.status = TaskStatus.Failed
            p.task_desc = f'upscaler task:{task.id} failed.'
        yield p

    def _save_images(self, task: Task, r: typing.List):
        local = []
        if r:
            name, _ = os.path.splitext(os.path.basename(task['image']))
            images = r[0]
            for image in images:
                full_path = os.path.join(Tmp, name + '.png')
                image.save(full_path)
                local.append(full_path)

        return local

    def _set_task_status(self, p: TaskProgress):
        super()._set_task_status(p)
        dumper.dump_task_progress(p)
