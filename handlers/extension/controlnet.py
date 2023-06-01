#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 10:34 AM
# @Author  : wangdongming
# @Site    :
# @File    : controlnet.py
# @Software: Hifive
import os.path
import time
import traceback
import typing
import numpy as np
import modules.scripts
from PIL import Image
from collections.abc import Iterable
from handlers.formatter import AlwaysonScriptArgsFormatter
from handlers.utils import get_tmp_local_path, Tmp, upload_files, strip_model_hash, upload_pil_image
from worker.task import TaskProgress, Task, TaskStatus, TaskType

ControlNet = 'ControlNet'
FreePreprocessors = [
    "reference_only",
    "reference_adain",
    "reference_adain+attn"
]
preprocessor_aliases = {
    "invert": "invert (from white bg & black line)",
    "lineart_standard": "lineart_standard (from white bg & black line)",
    "lineart": "lineart_realistic",
    "color": "t2ia_color_grid",
    "clip_vision": "t2ia_style_clipvision",
    "pidinet_sketch": "t2ia_sketch_pidi",
    "depth": "depth_midas",
    "normal_map": "normal_midas",
    "hed": "softedge_hed",
    "hed_safe": "softedge_hedsafe",
    "pidinet": "softedge_pidinet",
    "pidinet_safe": "softedge_pidisafe",
    "segmentation": "seg_ufade20k",
    "oneformer_coco": "seg_ofcoco",
    "oneformer_ade20k": "seg_ofade20k",
    "pidinet_scribble": "scribble_pidinet",
    "inpaint": "inpaint_global_harmonious",
}
reverse_preprocessor_aliases = {preprocessor_aliases[k]: k for k in preprocessor_aliases.keys()}


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


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
        self.image = np.array(Image.open(image).convert('RGB'))
        if not mask:
            shape = list(self.image.shape)
            if shape[-1] == 3:
                shape[-1] = 4  # rgba
            self.mask = np.zeros(self.image.shape)
            self.mask[:, :, -1] = 255
        else:
            mask = get_tmp_local_path(mask)
            self.mask = np.array(Image.open(mask).convert('RGBA'))
        module = strip_model_hash(module)
        module = reverse_preprocessor_aliases.get(module, module)

        if module == 'None':
            module = 'none'
        elif module in FreePreprocessors:
            module = 'none'
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
        scripts_runner = modules.scripts.scripts_img2img \
            if task.task_type == TaskType.Image2Image else modules.scripts.scripts_txt2img
        for script in scripts_runner.alwayson_scripts:
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

            module = args.module
            img = HWC3(args.image)
            if not ((args.mask[:, :, 0] == 0).all() or (args.mask[:, :, 0] == 255).all()):
                img = HWC3(args.mask[:, :, 0])

            if 'inpaint' in module:
                color = HWC3(args.image)
                alpha = args.mask[:, :, 0:1]
                img = np.concatenate([color, alpha], axis=2)

            # def get_module_basename(self, module):
            #     if module is None:
            #         module = 'none'
            #
            #     return global_state.reverse_preprocessor_aliases.get(module, module)

            preprocessor = control_net_script.preprocessor[args.module]
            # if pp:
            #     raw_H, raw_W, _ = img.shape
            #     target_H, target_W = t2i_h, t2i_w
            #     rm = str(rm)
            #
            #     k0 = float(target_H) / float(raw_H)
            #     k1 = float(target_W) / float(raw_W)
            #
            #     if rm == external_code.ResizeMode.OUTER_FIT.value:
            #         estimation = min(k0, k1) * float(min(raw_H, raw_W))
            #     else:
            #         estimation = max(k0, k1) * float(min(raw_H, raw_W))
            #
            #     pres = int(np.round(estimation))
            #     print(f'Pixel Perfect Mode Enabled In Preview.')
            #     print(f'resize_mode = {rm}')
            #     print(f'raw_H = {raw_H}')
            #     print(f'raw_W = {raw_W}')
            #     print(f'target_H = {target_H}')
            #     print(f'target_W = {target_W}')
            #     print(f'estimation = {estimation}')
            if args.pres > 64:
                result, is_image = preprocessor(img, res=args.pres, thr_a=args.pthr_a, thr_b=args.pthr_b)
            else:
                result, is_image = preprocessor(img)

            r, pli_img = None, None
            if is_image:
                if result.ndim == 3 and result.shape[2] == 4:
                    inpaint_mask = result[:, :, 3]
                    result = result[:, :, 0:3]
                    result[inpaint_mask > 127] = 0
                    pli_img = Image.fromarray(result, mode='RGB')
                elif result.ndim == 2:
                    pli_img = Image.fromarray(result, mode='L')
                else:
                    pli_img = Image.fromarray(result, mode='RGB')

            if pli_img:
                filename = task.id + '.png'
                r = upload_pil_image(True, pli_img, name=filename)

            progress = TaskProgress.new_finish(task, {
                'all': {
                    'high': [r]
                }
            })
            yield progress


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
                    image = Image.open(image).convert('RGBA')
                    size = image.size
                    image = np.array(image)
                    mask = item['image'].get('mask')
                    if not mask:
                        shape = list(size)
                        shape.append(4)  # rgba
                        mask = np.full(shape, 255)
                    else:
                        mask = get_tmp_local_path(item['image']['mask'])
                        mask = np.array(Image.open(mask))
                control_unit = {
                    'enabled': item.get('enabled', False),
                    'guess_mode': item.get('guess_mode', False),
                    'guidance_start': item.get('guidance_start', 0) or 0,
                    'guidance_end': item.get('guidance_end', 1) or 1,
                    'image': {
                        'image': image,
                        'mask': mask,
                    },
                    'invert_image': item.get('invert_image', False),
                    'low_vram': item.get('low_vram', False),
                    'model': item.get('model', 'none') or 'none',
                    'module': item.get('module', 'none') or 'none',
                    'processor_res': item.get('processor_res', 64) or 64,
                    'resize_mode': item.get('resize_mode', 'Crop and Resize') or 'Crop and Resize',
                    'rgbbgr_mode': item.get('rgbbgr_mode', False),
                    'threshold_a': item.get('threshold_a', 64) or 64,
                    'threshold_b': item.get('threshold_b', 64) or 64,
                    'weight': item.get('weight', 1) or 1,
                    'pixel_perfect': item.get('pixel_perfect', False),
                    'control_mode': item.get('control_mode', 'Balanced') or 'Balanced'
                }
                control_unit['module'] = strip_model_hash(control_unit['module'])
                control_unit['model'] = strip_model_hash(control_unit['model'])
                if control_unit['model'] == 'None':
                    control_unit['model'] = 'none'
                if control_unit['module'] == 'None':
                    control_unit['module'] = 'none'
                if control_unit['module'] in FreePreprocessors:
                    control_unit['model'] = 'none'

                new_args.append(control_unit)

            if isinstance(control_net_script_args, Iterable):
                for item in control_net_script_args:
                    set_default(item)

            control_net_script_args = new_args

        return control_net_script_args
