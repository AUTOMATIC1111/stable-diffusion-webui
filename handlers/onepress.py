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
from modules import shared, scripts
from enum import IntEnum
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus

from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path
from copy import deepcopy
import numpy as np
from typing import List, Union, Dict, Set, Tuple
import cv2
import numpy as np
from PIL import Image
import insightface
import onnxruntime
import os
import logging
from loguru import logger
import math
# conversion_action={"线稿":'line',"黑白":'black_white',"色块":'color','草图':'sketch','蜡笔':'crayon'}
# rendition_style={"彩铅":'color_pencil',"浮世绘":'ukiyo',"山水画":'landscape',"极简水彩":'min_watercolor',"炫彩":'dazzle_color',"油画":'oil_paint'}

def size_control(width, height):
    # 尺寸检查，最大1024*1024
    if width * height > 1024 * 1024:
        rate = (1024 * 1024) / (width * height)
        width = int(width * rate)
        height = int(height * rate)
    return width, height

def get_multidiffusion_args():
    onepress_multidiffusion_args = {'Tiled-Diffusion': {'args':
                                                            [{'batch_size': 1, 'causal_layers': False,
                                                              'control_tensor_cpu': False, 'controls':
                                                                  [{'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4},
                                                                   {'blend_mode': 'Background', 'enable': False,
                                                                    'feather_ratio': 0.2, 'h': 0.2, 'neg_prompt': '',
                                                                    'prompt': '', 'seed': -1, 'w': 0.2, 'x': 0.4,
                                                                    'y': 0.4}],

                                                              'draw_background': False, 'enable_bbox_control': False,
                                                              'enabled': True, 'image_height': 1024,
                                                              'image_width': 1024, 'keep_input_size': True,

                                                              'method': 'Mixture of Diffusers', 'noise_inverse': True,
                                                              'noise_inverse_renoise_kernel': 64,
                                                              'noise_inverse_renoise_strength': 0,

                                                              'noise_inverse_retouch': 1, 'noise_inverse_steps': 35,
                                                              'overlap': 48, 'overwrite_image_size': False,
                                                              'overwrite_size': False, 'scale_factor': 2,

                                                              'tile_height': 96, 'tile_width': 96,
                                                              'upscaler': 'None'}]}}

    return onepress_multidiffusion_args

def get_llul_args():
    onepress_llul_args = {'LLuL': {'args': [{'apply_to': ['OUT'], 'down': 'Pooling Max', 'down_aa': False,
                                             'enabled': True, 'force_float': False, 'intp': 'Lerp', 'layers':
                                                 'OUT', 'max_steps': 0, 'multiply': 1, 'start_steps': 1,
                                             'understand': False,

                                             'up': 'Bilinear', 'up_aa': False, 'weight': 0.15, 'x': 128, 'y': 128}]}}
    return onepress_llul_args

def get_cn_args():
    onepress_cn_args = {'control_mode': 'Balanced',
                        'enabled': False,
                        'guess_mode': False,
                        'guidance_end': 1,
                        'guidance_start': 0,
                        'image': {'image': None, 'mask': None},
                        'invert_image': False, 'low_vram': False, 'model': None, 'module': None,
                        'pixel_perfect': True,
                        'model': None,
                        'module': None,
                        'resize_mode': 'Scale to Fit (Inner Fit)',
                        'processor_res': 512,
                        'threshold_a': 1,
                        'threshold_b': -1,
                        'weight': 1}

    return onepress_cn_args

def get_txt2img_args(prompt, width, height):
    # 内置万能提示词
    prompt ='masterpiece, (highres 1.2), (ultra-detailed 1.2),'+prompt
    negative_prompt = '(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality,M,nsfw,'
    # 做尺寸控制
    width, height = size_control(width, height)
    onepress_txt2img_args = {'prompt': prompt, 'negative_prompt': negative_prompt,
                             'sampler_name': 'DPM++ 2M SDE Karras',
                             'steps': 35,
                             'cfg_scale': 7,
                             'denoising_strength': 0.7,
                             'width': width, 'height': height, 'n_iter': 1, 'batch_size': 1}
    return onepress_txt2img_args

def get_img2img_args(prompt, width, height, init_img):
    onepress_img2img_args = get_txt2img_args(prompt, width, height)
    onepress_img2img_args['denoising_strength'] = 0.75  # 蜡笔画
    onepress_img2img_args['init_img'] = init_img
    onepress_img2img_args['sketch'] = ''
    onepress_img2img_args['init_img_with_mask'] = {'image': '', 'mask': ''}
    return onepress_img2img_args

class OnePressTaskType(Txt2ImgTask):
    Conversion = 1  # 图片上色：
    Rendition = 2  # 风格转换

class ConversionTask(Txt2ImgTask):
    def __init__(self,
                 base_model_path: str,  # 模型路径
                 model_hash: str,  # 模型hash值
                 image: str,  # 图片路径
                 action: str,  # 转换方式
                 prompt: str,  # 图片的正向提示词
                 full_canny: bool = False,  # 图片细化
                 part_canny: bool = False,  # 区域细化
                 ):
        self.base_model_path = base_model_path
        self.model_hash = model_hash
        self.image = image
        self.action = action
        self.prompt = prompt
        self.full_canny = full_canny
        self.part_canny = part_canny

    @classmethod
    def exec_task(cls, task: Task):
        t = ConversionTask(
            task['base_model_path'],
            task['model_hash'],
            task['image'],
            task['action'],
            task['prompt'],
            task.get('full_canny', False),
            task.get('part_canny', False), )
        is_img2img = False
        full_canny = t.full_canny
        part_canny = t.part_canny
        full_task = deepcopy(task)
        full_task['alwayson_scripts'] = {'ControlNet': {'args': []}}

        """
        1.线稿-Linert，处理器-lineart_realistic（文生图）
        2.色块-tile，处理器-tile_colorfix+sharp（文生图）
        3.草图-canny，预处理器-canny（文生图）
        4.蜡笔-scrible，预处理器-scribble_pidinet（图生图模式）
        """
        # conversion_actiob={"线稿":'line',"黑白":'black_white',"色块":'color','草图':'sketch','蜡笔':'crayon'}
        if t.action == 'crayon':
            is_img2img = True
            init_img_inpaint = get_tmp_local_path(t.image)
            image = Image.open(init_img_inpaint).convert('RGB')
            full_task.update(get_img2img_args(t.prompt, image.width, image.height, t.image))
            cn_args = get_cn_args()
            cn_args['enabled'] = True
            cn_args['module'] = 'scribble_pidinet'
            cn_args['model'] = 'control_v11p_sd15_scribble [d4ba51ff]'
            cn_args['image']['image'] = t.image
            full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args)
        else:
            init_img_inpaint = get_tmp_local_path(t.image)
            image = Image.open(init_img_inpaint).convert('RGB')
            full_task.update(get_txt2img_args(t.prompt, image.width, image.height))
            if t.action == 'line':  # 线稿
                cn_args = get_cn_args()
                cn_args['enabled'] = True
                cn_args['module'] = 'invert (from white bg & black line)'
                cn_args['model'] = 'control_v11p_sd15_lineart [43d4be0d]'
                cn_args['image']['image'] = t.image
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args)
            elif t.action == 'black_white':  # 黑白
                cn_args_1 = get_cn_args()
                cn_args_1['enabled'] = True
                cn_args_1['module'] = 'lineart_realistic'
                cn_args_1['model'] = 'control_v11p_sd15_lineart [43d4be0d]'
                cn_args_1['image']['image'] = t.image
                cn_args_1['weight'] = 0.6
                cn_args_2 = get_cn_args()
                cn_args_2['enabled'] = True
                cn_args_2['module'] = 'depth_zoe'
                cn_args_2['model'] = 'diff_control_sd15_depth_fp16 [978ef0a1]'
                cn_args_2['image']['image'] = t.image
                cn_args_2['weight'] = 0.4
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args_1)
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args_2)
            elif t.action == 'sketch':  # 草图
                cn_args = get_cn_args()
                cn_args['enabled'] = True
                cn_args['module'] = 'canny'
                cn_args['model'] = 'control_v11p_sd15_canny [d14c016b]'
                cn_args['image']['image'] = t.image
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args)
            elif t.action == 'color':  # 色块
                cn_args = get_cn_args()
                cn_args['enabled'] = True
                cn_args['module'] = 'tile_colorfix+sharp'
                cn_args['model'] = 'control_v11f1e_sd15_tile [a371b31b]'
                cn_args['image']['image'] = t.image
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args)
            else:
                pass

        return full_task, is_img2img, full_canny, part_canny

class RenditionTask(Txt2ImgTask):
    def __init__(self,
                 base_model_path: str,  # 模型路径
                 model_hash: str,  # 模型hash值
                 style: str,  # 转绘风格 color_pencil/ukiyo/landscape/  min_watercolor/dazzle_color/oil_paint
                 width: int,  # 宽
                 height: int,  # 高
                 prompt: str,  # 图片的正向提示词
                 image: str,  # 原图路径
                 init_img: str,  # 油画垫图
                 lora_models: typing.Sequence[str] = None,# lora 
                 roop: bool = False,  # 换脸
                 batch_size:int=1 #结果数量
                 ):
        self.base_model_path = base_model_path
        self.model_hash = model_hash
        self.loras = lora_models
        self.prompt = prompt
        self.width = width if width!=0 else 512
        self.height = height if height!=0 else 512
        self.style = style
        self.image = image
        self.init_img = init_img
        self.roop = roop
        self.batch_size=batch_size if batch_size!=0 else 1

    @classmethod
    def exec_task(cls, task: Task):

        t = RenditionTask(
            task['base_model_path'],
            task['model_hash'],
            task['style'],
            task['width'],
            task['height'],
            task.get('prompt',""),
            task.get('image', None), # 图生图：极简水彩 炫彩 油画
            task.get('init_img', None),# 风格垫图 油画判断
            task.get('lora_models',None),
            task.get('roop', False), 
            task.get('batch_size',1))
        full_task = deepcopy(task)

        # 根据style的不同 把contorlnet的参数值加进去
        full_task.update(get_txt2img_args(t.prompt, t.width, t.height))
        full_task['prompt']=t.prompt
        full_task['batch_size']=t.batch_size
        full_task['alwayson_scripts'] = {'ControlNet': {'args': []}}
        # rendition_style={"彩铅":'color_pencil',"浮世绘":'ukiyo',"山水画":'landscape',"极简水彩":'min_watercolor',"炫彩":'dazzle_color',"油画":'oil_paint'}
        if t.style in ['min_watercolor', 'dazzle_color', 'oil_paint']:
            # 线稿固定人物形态
            cn_args_1 = get_cn_args()
            cn_args_1['enabled'] = True
            cn_args_1['module'] = 'invert (from white bg & black line)' # if t.style!='oil_paint'else 'lineart_realistic'
            cn_args_1['model'] = 'control_v11p_sd15_lineart [43d4be0d]'
            cn_args_1['image']['image'] = t.image
            full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args_1)
            # 极简水彩：step：10,cfg：3
            if t.style=='min_watercolor':
                full_task['steps']=10
                full_task['cfg_scale']=3
            # 油画：风格垫图
            if t.style=='oil_paint':
                full_task['sampler_name']='DPM++ 2M Karras'
                cn_args_2 = get_cn_args()
                cn_args_2['enabled'] = True
                cn_args_2['module'] = 'reference_only'
                cn_args_2['image']['image'] = t.init_img # 风格垫图
                cn_args_2['weight'] = 1
                cn_args_2['threshold_a']=0.5
                cn_args_2['threshold_b']=0
                full_task['alwayson_scripts']['ControlNet']['args'].append(cn_args_2)
        else:
            pass
        if t.style=='color_pencil': # 彩铅
                full_task['sampler_name']='DPM++ 2S a Karras'
                full_task['steps']=20
        if full_task['lora_models']==['']:full_task['lora_models']=None
        return full_task

    @classmethod
    def exec_roop(cls,source_img: Image.Image, target_img: Image.Image, ):

        def get_face(img_data: np.ndarray,providers, det_size=(640, 640)):
            # TODO 更换路径
            models_dir = os.path.join(scripts.basedir(), "models" + os.path.sep + "roop"+os.path.sep + "buffalo_l")
            face_analyser = insightface.app.FaceAnalysis(name=models_dir, providers=providers)
            face_analyser.prepare(ctx_id=0, det_size=det_size)
            face = face_analyser.get(img_data)

            if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
                det_size_half = (det_size[0] // 2, det_size[1] // 2)
                return get_face(img_data, providers,det_size=det_size_half)

            try:
                return sorted(face, key=lambda x: x.bbox[0])
            except IndexError:
                return None

        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        
        # 获取人脸
        providers = onnxruntime.get_available_providers()
        source_face = get_face(source_img,providers)
        target_face = get_face(target_img,providers)
        if source_face is None or target_face is None: 
            return target_img
        # 人脸对应：算出中心点，判别与目标点最近的一张脸的位置，进行交换
        source_point=[]
        for _ in source_face:
            box=_.bbox.astype(np.int)
            source_point.append(((box[2]+box[0])/2,(box[3]+box[1])/2))
        
        # 交换人脸
        models_path = os.path.join(scripts.basedir(), "models" + os.path.sep + "roop" + os.path.sep + "inswapper_128.onnx")
        face_swapper = insightface.model_zoo.get_model(models_path, providers=providers)
        result = target_img
        for _ in target_face:
            box=_.bbox.astype(np.int)
            point=((box[2]+box[0])/2,(box[3]+box[1])/2)
            dis=[math.sqrt((point[0]-k[0])**2 + (point[1]-k[1])**2) for k in source_point]
             # 距离最近的人脸
            s_face=source_face[dis.index(min(dis))]
            result = face_swapper.get(result, _, s_face)
        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        # 人脸修复的过程
        original_image = result_image.copy()
        numpy_image = np.array(result_image)
        face_restorer = None
        for restorer in shared.face_restorers:
            if restorer.name() == 'GFPGAN':
                face_restorer = restorer
                break
        numpy_image = face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(original_image, restored_image, 1)

        return result_image


class OnePressTaskHandler(Txt2ImgTaskHandler):
    def __init__(self):
        super(OnePressTaskHandler, self).__init__()
        self.task_type = TaskType.OnePress

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        # 根据任务的不同类型：执行不同的任务
        if task.minor_type == OnePressTaskType.Conversion:
            # yield from self._exec_rendition(task)
            yield from self._exec_conversion(task)
        if task.minor_type == OnePressTaskType.Rendition:
            yield from self._exec_rendition(task)

    def _build_gen_canny_i2i_args(self, t, processed: Processed):
        denoising_strength = 0.5
        cfg_scale = 7

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
            n_iter=1,
            steps=t.steps,
            cfg_scale=cfg_scale,
            width=t.width,
            height=t.height,
            restore_faces=t.restore_faces,
            tiling=t.tiling,
            init_images=processed.images,
            mask=None,
            denoising_strength=denoising_strength
        )

    def _canny_process_i2i(self, p, alwayson_scripts):
        fix_seed(p)

        images = p.init_images

        save_normally = True

        p.do_not_save_grid = True
        p.do_not_save_samples = not save_normally

        shared.state.job_count = len(images) * p.n_iter

        image = images[0]
        shared.state.job = f"{1} out of {len(images)}"

        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(image)
        p.init_images = [img] * p.batch_size

        i2i_script_runner = modules.scripts.scripts_img2img
        default_script_arg_img2img = init_default_script_args(modules.scripts.scripts_img2img)
        selectable_scripts, selectable_script_idx = get_selectable_script(i2i_script_runner, None)
        select_script_args = ''
        script_args = init_script_args(default_script_arg_img2img, alwayson_scripts, selectable_scripts,
                                       selectable_script_idx, select_script_args, i2i_script_runner)
        script_args = tuple(script_args)

        p.scripts = i2i_script_runner
        p.script_args = script_args
        proc = modules.scripts.scripts_img2img.run(p, *script_args)
        if proc is None:
            proc = process_images(p)
        # 只返回第一张
        proc.images = [proc.images[0]]
        return proc

    def _exec_conversion(self, task: Task) -> typing.Iterable[TaskProgress]:
        logger.info("one press conversion func starting...")
        full_task, is_img2img, full_canny, part_canny = ConversionTask.exec_task(task)
        logger.info("download model...")
        # 加载模型
        base_model_path = self._get_local_checkpoint(full_task)
        logger.info(f"load model:{base_model_path}")
        load_sd_model_weights(base_model_path, full_task.model_hash)

        # 第一阶段 上色
        progress = TaskProgress.new_ready(full_task, f'model loaded, run onepress_paint...')
        yield progress

        logger.info("download network models...")
        process_args = self._build_img2img_arg(progress) if is_img2img else self._build_txt2img_arg(progress)
        self._set_little_models(process_args)
        progress.status = TaskStatus.Running
        progress.task_desc = f'onepress task({task.id}) running'
        yield progress

        logger.info("step 1, colour...")
        shared.state.begin()
        processed = process_images(process_args)
        logger.info("step 1 > ok")
        # 第二阶段 细化
        if part_canny or full_canny:
            alwayson_scripts = {}
            if part_canny:
                alwayson_scripts.update(get_llul_args())
            if full_canny:
                alwayson_scripts.update(get_multidiffusion_args())
            # i2i
            logger.info("step 2, canny...")
            processed_i2i_1 = self._build_gen_canny_i2i_args(process_args, processed)
            processed_i2i = self._canny_process_i2i(processed_i2i_1, alwayson_scripts)
            processed.images = processed_i2i.images
        logger.info("step 2 > ok")
        shared.state.end()
        process_args.close()

        logger.info("step 3, upload images...")
        progress.status = TaskStatus.Uploading
        yield progress

        # 只返回最终结果图
        processed.images = [processed.images[0]]
        images = save_processed_images(processed,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       task.id,
                                       inspect=process_args.kwargs.get("need_audit", False))
        logger.info("step 3 > ok")
        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed.all_seeds, processed.all_subseeds)

        yield progress

    def _exec_rendition(self, task: Task) -> typing.Iterable[TaskProgress]:
        
        logger.info("one press rendition func starting...")

        full_task = RenditionTask.exec_task(task)
        
        # 加载模型
        base_model_path = self._get_local_checkpoint(full_task)
        logger.info("download model...")
        load_sd_model_weights(base_model_path, full_task.model_hash)
        progress = TaskProgress.new_ready(full_task, f'model loaded, run onepress_paint...')
        yield progress

        logger.info("download network models...")
        process_args =  self._build_txt2img_arg(progress)

        # 加载lora模型
        self._set_little_models(process_args)
        progress.status = TaskStatus.Running
        progress.task_desc = f'onepress task({task.id}) running'
        yield progress

        logger.info("step 1, rendition...")
        shared.state.begin()
        processed = process_images(process_args)
        processed.images = processed.images[1:task['batch_size']+1]
        logger.info("step 1 > ok")

        # 如果需要换人脸
        if task['roop']:
            logger.info("step 2, roop...")
            roop_result=[]
            for i,img in enumerate(processed.images):
                source_img = get_tmp_local_path(task['image'])
                source_img = Image.open(source_img).convert('RGB')
                target_img=img # 
                # source_img: Image.Image, target_img: Image.Image, 
                roop_result.append(RenditionTask.exec_roop(source_img,target_img))
            processed.images=roop_result 
            logger.info("step 2 > ok")

        shared.state.end()
        process_args.close()

        logger.info("step 3, upload images...")
        progress.status = TaskStatus.Uploading
        yield progress
        images = save_processed_images(processed,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       task.id,
                                       inspect=process_args.kwargs.get("need_audit", False))
        logger.info("step 3 > ok")
        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed.all_seeds, processed.all_subseeds)
        yield progress