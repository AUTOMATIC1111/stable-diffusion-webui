#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 8:45 PM
# @Author  : wangdongming
# @Site    : 
# @File    : extra.py
# @Software: Hifive
import os.path
import random
import typing
from PIL import Image
from enum import IntEnum
from worker.dumper import dumper
from modules.postprocessing import run_extras
from handlers.typex import ImageKeys
from tools.image import compress_image
from worker.handler import DumpTaskHandler
from handlers.utils import get_tmp_local_path, Tmp, upload_files
from worker.task import Task, TaskType, TaskProgress, TaskStatus


class ExtraTaskMinorTaskType(IntEnum):
    Default = 0
    SingleUpscaler = 1


class SingleUpscalerTask:

    def __init__(self,
                 image: str,  # 图片路径
                 resize_mode: int = 0,  # 模式 0-scale by 1-scale to
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
        self.resize_mode = resize_mode
        self.extras_mode = 0  # 0-single, 1-batch
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
            self.image = Image.open(local_image).convert('RGB')
        else:
            raise OSError(f'cannot found image:{image}')

    @classmethod
    def exec_task(cls, task: Task):
        t = SingleUpscalerTask(
            task['image'],
            task.get('resize_mode', 0),
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


class ExtraTaskHandler(DumpTaskHandler):

    def __init__(self):
        super(ExtraTaskHandler, self).__init__(TaskType.Extra)

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        minor_type = ExtraTaskMinorTaskType(task.minor_type)
        if minor_type <= ExtraTaskMinorTaskType.SingleUpscaler:
            yield from self.__exec_single_upscaler_task(task)

    def __exec_single_upscaler_task(self, task: Task):
        p = TaskProgress.new_running(task, "up scale image...")
        yield p
        result = SingleUpscalerTask.exec_task(task)
        p.task_progress = random.randint(30, 70)
        yield p
        if result:
            high, low = self._save_images(task, result)
            p.task_progress = random.randint(70, 90)
            yield p
            high_keys = upload_files(False, *high)
            low_keys = upload_files(False, *low)
            image_keys = ImageKeys(high_keys, low_keys)
            images = result[0]
            # p.set_finish_result({
            #     'all': image_keys.to_dict(),
            #     'upscaler': {
            #         'size': '' if not images else f'{images[0].width}*{images[0].height}',
            #     }
            # })
            p = TaskProgress.new_finish(task,
                                        {
                                            'all': image_keys.to_dict(),
                                            'upscaler': {
                                                'size': '' if not images else f'{images[0].width}*{images[0].height}',
                                            }
                                        })
            p.task_desc = f'upscaler task:{task.id} finished.'
        else:
            p.status = TaskStatus.Failed
            p.task_desc = f'upscaler task:{task.id} failed.'
        yield p

    def _save_images(self, task: Task, r: typing.List):
        high, low = [], []
        if r:
            images = r[0]
            for image in images:
                filename = task.id + '.png'
                full_path = os.path.join(Tmp, filename)
                image.save(full_path)
                high.append(full_path)
                low_file = os.path.join(Tmp, 'low-' + filename)
                if not os.path.isfile(low_file):
                    compress_image(full_path, low_file)
                low.append(low_file)

        return high, low
