#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 10:34 AM
# @Author  : wangdongming
# @Site    :
# @File    : controlnet.py
# @Software: Hifive

import time
import traceback
import typing
import numpy as np
from PIL import Image
from collections.abc import Iterable
from modules.scripts import scripts_img2img
from handlers.formatter import AlwaysonScriptArgsFormatter
from handlers.utils import get_tmp_local_path
from worker.task import TaskHandler, TaskType, TaskProgress, Task, TaskStatus

ControlNet = 'ControlNet'


class RunAnnotatorArgs:

    def __init__(self,
                 image: str,  # 图片路径
                 mask: str,  # 蒙版路径
                 module: str,  # 预处理器
                 annotator_resolution: int = 512,  # 分辨率
                 pthr_a: int = 64,  # 阈值A
                 pthr_b: int = 64,  # 阈值B
                 **kwargs):
        image = get_tmp_local_path(image)
        self.image = np.array(Image.open(image))
        if not mask:
            shape = list(self.image.shape)
            if shape[-1] == 3:
                shape[-1] = 4  # rgba
            self.mask = np.zeros(self.image.shape)
            self.mask[:, :, -1] = 255
        else:
            mask = get_tmp_local_path(mask)
            self.mask = np.array(Image.open(mask))
        self.kwargs = kwargs
        self.module = module
        self.pres = annotator_resolution
        self.pthr_a = pthr_a
        self.pthr_b = pthr_b


def build_run_annotato_args(task: Task) -> typing.Tuple[typing.Optional[RunAnnotatorArgs], str]:
    try:
        args = RunAnnotatorArgs(**task)
        return args, ''
    except Exception as err:
        return None, traceback.format_exc()


def exec_control_net_annotator(task: Task) -> typing.Iterable[TaskProgress]:
    progress = TaskProgress.new_ready(task, 'at the ready')
    yield progress
    args, err = build_run_annotato_args(task)
    if not args:
        progress.status = TaskStatus.Failed
        progress.task_desc = 'arg err:' + err
        yield progress
    else:
        control_net_script = None
        for script in scripts_img2img.alwayson_scripts:
            if 'ControlNet' != script.title():
                continue
            control_net_script = script
        if not control_net_script:
            progress.status = TaskStatus.Failed
            progress.task_desc = 'cannot found controlnet script'
            yield progress
        else:
            progress.status = TaskStatus.Running
            progress.task_desc = 'run annotator'
            yield progress
            # control_net_script.run_annotator(**args.args)

            img = control_net_script.HWC3(args.image)
            if not ((args.mask[:, :, 0] == 0).all() or (args.mask[:, :, 0] == 255).all()):
                img = control_net_script.HWC3(args.mask[:, :, 0])
            preprocessor = control_net_script.preprocessor[args.module]
            if args.pres > 64:
                result, is_image = preprocessor(img, res=args.pres, thr_a=args.pthr_a, thr_b=args.pthr_b)
            else:
                result, is_image = preprocessor(img)
            if is_image:
                pass


def bind_debug_img_task_args(*tasks: Task):
    test_img = 'test-imgs/QQ20230316-184425.png'
    alwayson_scripts = {}
    alwayson_scripts['ControlNet'] = {
        'args': [
            {
                'image': {
                    'image': test_img,
                },
                'model': 'control_openpose-fp16 [9ca67cc5]',
                'module': 'openpose_hand',
                'enabled': True,
            }
        ]
    }

    for t in tasks:
        t['alwayson_scripts'] = alwayson_scripts
        yield t


class ControlnetFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return ControlNet

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        if isinstance(args, dict):
            # 只传了一个ControlNetUnit对象，转换为LIST处理
            args = [args]
        control_net_script_args = [x for x in args]
        if control_net_script_args:
            new_args = []

            def set_default(item):
                image, mask = None, None
                if item.get('enabled', False):
                    image = get_tmp_local_path(item['image']['image'])
                    image = np.array(Image.open(image))
                    mask = item['image'].get('mask')
                    if not mask:
                        shape = list(image.shape)
                        if shape[-1] == 3:
                            shape[-1] = 4  # rgba
                        mask = np.zeros(image.shape)
                        mask[:, :, -1] = 255
                    else:
                        mask = get_tmp_local_path(item['image']['mask'])
                        mask = np.array(mask)
                new_args.append({
                    'enabled': item.get('enabled', False),
                    'guess_mode': item.get('guess_mode', False),
                    'guidance_start': item.get('guidance_start', 0),
                    'guidance_end': item.get('guidance_end', 1),
                    'image': {
                        'image': image,
                        'mask': mask,
                    },
                    'invert_image': item.get('invert_image', False),
                    'low_vram': item.get('low_vram', False),
                    'model': item.get('model', 'None'),
                    'module': item.get('module', 'none'),
                    'processor_res': item.get('processor_res', 64),
                    'resize_mode': item.get('resize_mode', 'Scale to Fit (Inner Fit)'),
                    'rgbbgr_mode': item.get('rgbbgr_mode', False),
                    'threshold_a': item.get('threshold_a', 64),
                    'threshold_b': item.get('threshold_b', 64),
                    'weight': item.get('weight', 1),
                })

            if isinstance(control_net_script_args, Iterable):
                for item in control_net_script_args:
                    set_default(item)

            control_net_script_args = new_args

        return control_net_script_args
