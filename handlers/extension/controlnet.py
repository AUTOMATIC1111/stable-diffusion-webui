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
annotato_args_thr_a_dict = {
    'canny': [1, 255],
    'depth_leres': [0, 100],
    'depth_leres++': [0, 100],
    'mediapipe_face': [1, 10],
    'mlsd': [0.01, 2],
    'normal_midas': [0, 1],
    'reference_adain': [0, 1],
    'reference_adain+attn': [0, 1],
    'reference_only': [0, 1],
    'scribble_xdog': [1, 64],
    'threshold': [0, 255],
    'tile_colorfix': [3, 32],
    'tile_colorfix+sharp': [3, 32],
    'tile_resample': [1, 8]}
annotato_args_thr_b_dict = {
    'canny': [1, 255],
    'depth_leres': [0, 100],
    'depth_leres++': [0, 100],
    'mediapipe_face': [1, 10],
    'mlsd': [0.01, 20],
    'tile_colorfix+sharp': [0, 2]}
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


def clip_vision_visualization(x):
    x = x.detach().cpu().numpy()[0]
    x = np.ascontiguousarray(x).copy()
    return np.ndarray((x.shape[0] * 4, x.shape[1]), dtype="uint8", buffer=x.tobytes())


class RunAnnotatorArgs:

    def __init__(self,
                 image: str,  # 图片路径
                 mask: str,  # 蒙版路径
                 module: str,  # 预处理器
                 annotator_resolution: int = 512,  # 分辨率
                 pthr_a: int = 64,  # 阈值A
                 pthr_b: int = 64,  # 阈值B
                 t2i_w: int = 1080,
                 t2i_h: int = 1520,
                 pp: bool = False,
                 rm: str = 'Resize and Fill',
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
        self.module = module
        self.pres = annotator_resolution
        self.pthr_a = pthr_a
        self.pthr_b = pthr_b
        self.t2i_w = t2i_w
        self.t2i_h = t2i_h
        self.pp = pp
        self.rm = rm
        self.kwargs = kwargs


def build_run_annotato_args(task: Task) -> typing.Tuple[typing.Optional[RunAnnotatorArgs], str]:
    try:
        args = RunAnnotatorArgs(**task)
        return args, ''
    except Exception as err:
        return None, traceback.format_exc()


def run_annotato_args_check(module, thr_a, thr_b):
    thr_a = thr_a
    thr_b = thr_b
    if module in annotato_args_thr_a_dict.keys():
        if not min(annotato_args_thr_a_dict[module]) <= thr_a <= max(annotato_args_thr_a_dict[module]):
            thr_a = min(annotato_args_thr_a_dict[module])
    if module in annotato_args_thr_b_dict.keys():
        if not min(annotato_args_thr_b_dict[module]) <= thr_b <= max(annotato_args_thr_b_dict[module]):
            thr_b = min(annotato_args_thr_b_dict[module])
    return thr_a, thr_b


def pixel_perfect_resolution(
        image: np.ndarray,
        target_H: int,
        target_W: int,
        resize_mode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == "Resize and Fill":
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    return int(np.round(estimation))


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
            if args.pp:
                args.pres = pixel_perfect_resolution(
                    img,
                    target_H=args.t2i_h,
                    target_W=args.t2i_w,
                    resize_mode=args.rm,
                )
            if args.pres > 64:
                # 参数校验：超过范围就取最小值
                args.pthr_a, args.pthr_b = run_annotato_args_check(args.module, args.pthr_a, args.pthr_b)
                result, is_image = preprocessor(img, res=args.pres, thr_a=args.pthr_a, thr_b=args.pthr_b)
            else:
                result, is_image = preprocessor(img)

            if "clip" in module:
                result = clip_vision_visualization(result)
                is_image = True
                
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
                    image = get_tmp_local_path(item['image']['image']) if item['image']['image'] else None
                    image = Image.open(image).convert('RGBA') if image else None
                    size = image.size if image else None
                    image = np.array(image) if image else None
                    mask = item['image'].get('mask')
                    if not mask:
                        shape = list(size)
                        shape.append(4)  # rgba
                        mask = np.zeros(shape)
                        mask[:, :, -1] = 255
                        # mask = None
                    elif isinstance(mask, str) and mask:
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
                    'processor_res': item.get('processor_res', 64),
                    'resize_mode': item.get('resize_mode', 'Crop and Resize') or 'Crop and Resize',
                    'rgbbgr_mode': item.get('rgbbgr_mode', False),
                    'threshold_a': item.get('threshold_a', 64),
                    'threshold_b': item.get('threshold_b', 64),
                    'weight': item.get('weight', 1) or 1,
                    'pixel_perfect': item.get('pixel_perfect', False),
                    'control_mode': item.get('control_mode', 'Balanced') or 'Balanced'
                }
                # 参数校验
                control_unit['threshold_a'], control_unit['threshold_b'] = run_annotato_args_check(
                    control_unit['module'], control_unit['threshold_a'], control_unit['threshold_b'])

                control_unit['module'] = strip_model_hash(control_unit['module'])
                control_unit['model'] = strip_model_hash(control_unit['model'])
                # if control_unit['model'] == 'None':
                #     control_unit['model'] = 'none'
                if control_unit['module'] == 'None':
                    control_unit['module'] = 'none'
                if control_unit['module'] in FreePreprocessors:
                    control_unit['model'] = 'None'

                new_args.append(control_unit)

            if isinstance(control_net_script_args, Iterable):
                for item in control_net_script_args:
                    set_default(item)

            control_net_script_args = new_args

        return control_net_script_args
