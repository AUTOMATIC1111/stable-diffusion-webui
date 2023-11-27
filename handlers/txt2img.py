#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/19 9:41 AM
# @Author  : wangdongming
# @Site    : 
# @File    : txt2img.py
# @Software: Hifive
import os
import time
import typing
import modules.scripts
import modules.shared as shared
from enum import IntEnum
from handlers.typex import ModelType
from modules.generation_parameters_copypaste import create_override_settings_dict
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingTxt2Img, process_images, Processed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, format_override_settings
from handlers.extension.controlnet import exec_control_net_annotator
from handlers.img2img import Img2ImgTaskHandler, AlwaysonScriptsType


class Txt2ImgMinorTaskType(IntEnum):
    Default = 0
    Txt2Img = 1
    RunControlnetAnnotator = 100


class Txt2ImgTask(StableDiffusionProcessingTxt2Img):

    def __init__(self, base_model_path: str,
                 user_id: str,
                 default_script_arg_txt2img: typing.Sequence,  # 默认脚本参数，handler构造。
                 prompt: str,  # TAG
                 negative_prompt: str,  # 反向TAG
                 sampler_name: str = None,  # 采样器
                 enable_hr: bool = False,
                 denoising_strength: float = 0.7,  # 重绘幅度
                 tiling: bool = False,  # 可平铺
                 hr_scale: float = 2.0,
                 hr_upscaler: str = None,
                 hr_second_pass_steps: int = 0,
                 hr_resize_x: int = 0,
                 hr_resize_y: int = 0,
                 cfg_scale: float = 7.0,  # 提示词相关性
                 width: int = 512,  # 图宽
                 height: int = 512,  # 图高
                 restore_faces: bool = False,  # 面部修复
                 seed: int = -1,  # 随机种子
                 seed_enable_extras: bool = False,  # 是否启用随机种子扩展
                 subseed: int = -1,  # 差异随机种子
                 subseed_strength: float = 0,  # 差异强度
                 seed_resize_from_h: int = 0,  # 重置尺寸种子-高度
                 seed_resize_from_w: int = 0,  # 重置尺寸种子-宽度
                 batch_size: int = 1,  # 批次数量
                 n_iter: int = 1,  # 每个批次数量
                 steps: int = 30,  # 步数
                 select_script_name: str = None,  # 选择下拉框脚本名
                 select_script_args: typing.Sequence = None,  # 选择下拉框脚本参数
                 select_script_nets: typing.Sequence[typing.Mapping] = None,  # 选择下拉框脚本涉及的模型信息
                 alwayson_scripts: AlwaysonScriptsType = None,  # 插件脚本，object格式： {插件名: {'args': [参数列表]}}
                 override_settings_texts=None,  # 自定义设置 TEXT,如: ['Clip skip: 2', 'ENSD: 31337', 'sd_vae': 'None']
                 lora_models: typing.Sequence[str] = None,  # 使用LORA，用户和系统全部LORA列表
                 embeddings: typing.Sequence[str] = None,  # embeddings，用户和系统全部mbending列表
                 lycoris_models: typing.Sequence[str] = None,  # lycoris，用户和系统全部lycoris列表
                 compress_pnginfo: bool = True,  # 使用GZIP压缩图片信息（默认开启）
                 hr_sampler_name: str = None,  # hr sampler
                 hr_prompt: str = None,  # hr prompt
                 hr_negative_prompt: str = None,  # hr negative prompt
                 disable_ad_face: bool = False,  # 关闭默认的ADetailer face
                 enable_refiner: bool = False,  # 是否启用XLRefiner
                 refiner_switch_at: float = 0.2,  # XL 精描切换时机
                 refiner_checkpoint: str = None,  # XL refiner模型文件
                 **kwargs):
        # "upcast_attn": false # 将交叉关注层向上转型到float32
        #
        # fast模式下关闭默认的AD插件
        disable_ad_face = disable_ad_face or kwargs.get('is_fast', False)
        override_settings_texts = format_override_settings(override_settings_texts)
        override_settings = create_override_settings_dict(override_settings_texts)

        t2i_script_runner = modules.scripts.scripts_txt2img
        selectable_scripts, selectable_script_idx = get_selectable_script(t2i_script_runner, select_script_name)
        script_args = init_script_args(default_script_arg_txt2img, alwayson_scripts, selectable_scripts,
                                       selectable_script_idx, select_script_args, t2i_script_runner,
                                       not disable_ad_face, enable_refiner, refiner_switch_at, refiner_checkpoint,
                                       seed, seed_enable_extras, subseed, subseed_strength, seed_resize_from_h,
                                       seed_resize_from_w)

        self.sd_model = shared.sd_model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.seed_enable_extras = seed_enable_extras
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.width = width
        self.height = height
        self.restore_faces = restore_faces
        self.tiling = tiling
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength if enable_hr else None
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.override_settings = override_settings
        self.outpath_samples = f"output/{user_id}/txt2img/samples/"
        self.outpath_grids = f"output/{user_id}/txt2img/grids/"
        self.outpath_scripts = f"output/{user_id}/img2img/scripts/"
        self.hr_sampler_name = hr_sampler_name
        self.hr_prompt = hr_prompt or ""
        self.hr_negative_prompt = hr_negative_prompt or ""
        self.scripts = modules.scripts.scripts_txt2img
        self.script_args = script_args
        self.script_name = select_script_name
        self.base_model_path = base_model_path
        self.selectable_scripts = selectable_scripts
        self.compress_pnginfo = compress_pnginfo
        self.kwargs = kwargs
        self.loras = lora_models
        self.embedding = embeddings
        self.lycoris = lycoris_models
        self.select_script_nets = select_script_nets
        self.xl_refiner = enable_refiner
        self.refiner_switch_at = refiner_switch_at
        self.xl_refiner_model_path = refiner_checkpoint

        super(Txt2ImgTask, self).__post_init__()

    def close(self):
        for obj in self.script_args:
            if hasattr(obj, 'close'):
                obj.close()
            if isinstance(obj, dict):
                for v in obj.values():
                    if hasattr(v, 'close'):
                        v.close()

    @classmethod
    def from_task(cls, task: Task, default_script_args: typing.Sequence, refiner_checkpoint: str = None):
        base_model_path = task['base_model_path']
        alwayson_scripts = task['alwayson_scripts']
        user_id = task['user_id']
        select_script = task.get('select_script')
        select_script_name, select_script_args = None, None
        prompt = task.get('prompt', '')
        negative_prompt = task.get('negative_prompt', '')

        if select_script:
            if not isinstance(select_script, dict):
                raise TypeError('select_script type err')
            select_script_name = select_script['name']
            select_script_args = select_script['args']
        else:
            select_script_name = task.get('select_script_name')
            select_script_args = task.get('select_script_args')
        kwargs = task.data.copy()
        kwargs.pop('base_model_path')
        kwargs.pop('alwayson_scripts')
        kwargs.pop('prompt')
        kwargs.pop('negative_prompt')
        kwargs.pop('user_id')
        if 'select_script' in kwargs:
            kwargs.pop('select_script')
        if 'select_script_name' in kwargs:
            kwargs.pop('select_script_name')
        if 'select_script_args' in kwargs:
            kwargs.pop('select_script_args')

        if "nsfw" in prompt.lower():
            prompt = prompt.lower().replace('nsfw', '')
        kwargs['refiner_checkpoint'] = refiner_checkpoint

        return cls(base_model_path,
                   user_id,
                   default_script_args,
                   prompt=prompt,
                   negative_prompt=negative_prompt,
                   alwayson_scripts=alwayson_scripts,
                   select_script_name=select_script_name,
                   select_script_args=select_script_args,
                   **kwargs)


class Txt2ImgTaskHandler(Img2ImgTaskHandler):

    def __init__(self):
        super(Txt2ImgTaskHandler, self).__init__()
        self.task_type = TaskType.Txt2Image

    def _load_default_script_args(self):
        self.default_script_args = init_default_script_args(modules.scripts.scripts_txt2img)
        self._default_script_args_load_t = time.time()

    def _build_txt2img_arg(self, progress: TaskProgress, refiner_checkpoint: str = None) -> Txt2ImgTask:
        self._refresh_default_script_args()
        t = Txt2ImgTask.from_task(progress.task, self.default_script_args, refiner_checkpoint)
        shared.state.current_latent_changed_callback = lambda: self._update_preview(progress)
        return t

    # def _get_local_checkpoint(self, task: Task):
    #     progress = TaskProgress.new_prepare(task, f"0%")
    #     xl_refiner_model_path = task.get('xl_refiner_model_path')
    #     # 脚本任务
    #     self._get_select_script_models(progress)
    #
    #     def base_model_progress_callback(*args):
    #         if len(args) < 2:
    #             return
    #         transferred, total = args[0], args[1]
    #         p = int(transferred * 100 / total)
    #         if xl_refiner_model_path:
    #             p = p * 0.5
    #
    #         current_progress = int(progress.task_desc[:-1])
    #         if p % 5 == 0 and p >= current_progress + 5:
    #             progress.task_desc = f"{p}%"
    #             self._set_task_status(progress)
    #
    #     base_model_path = get_model_local_path(task.sd_model_path, ModelType.CheckPoint, base_model_progress_callback)
    #     if not base_model_path or not os.path.isfile(base_model_path):
    #         raise OSError(f'cannot found model:{task.sd_model_path}')
    #
    #     def refiner_model_progress_callback(*args):
    #         if len(args) < 2:
    #             return
    #         transferred, total = args[0], args[1]
    #         p = int(50 + transferred * 100 * 0.5 / total)
    #
    #         current_progress = int(progress.task_desc[:-1])
    #         if p % 5 == 0 and p >= current_progress + 5:
    #             progress.task_desc = f"{p}%"
    #             self._set_task_status(progress)
    #
    #     xl_refiner_model = get_model_local_path(
    #         xl_refiner_model_path, ModelType.CheckPoint, refiner_model_progress_callback)
    #     if not xl_refiner_model or not os.path.isfile(xl_refiner_model):
    #         raise OSError(f'cannot found model:{xl_refiner_model_path}')
    #
    #     return base_model_path, xl_refiner_model

    def refiner_image(self, xl_refiner_model, images):
        sha256, _ = os.path.splitext(os.path.basename(xl_refiner_model))
        load_sd_model_weights(xl_refiner_model, sha256)
        # todo: batch exec refiner images

    def _exec_txt2img(self, task: Task) -> typing.Iterable[TaskProgress]:
        local_model_paths = self._get_local_checkpoint(task)
        base_model_path = local_model_paths if not isinstance(local_model_paths, tuple) else local_model_paths[0]
        refiner_checkpoint = None if not isinstance(local_model_paths, tuple) else local_model_paths[1]

        load_sd_model_weights(base_model_path, task.model_hash)
        progress = TaskProgress.new_ready(task, f'model loaded, run t2i...')
        yield progress
        process_args = self._build_txt2img_arg(progress, refiner_checkpoint)
        self._set_little_models(process_args)
        progress.status = TaskStatus.Running
        progress.task_desc = f't2i task({task.id}) running'
        yield progress
        shared.state.begin()
        # shared.state.job_count = process_args.n_iter * process_args.batch_size
        inference_start = time.time()
        if process_args.selectable_scripts:
            processed = process_args.scripts.run(process_args, *process_args.script_args)
        else:
            processed = process_images(process_args)
        shared.state.end()
        process_args.close()
        inference_time = time.time() - inference_start
        progress.status = TaskStatus.Uploading
        yield progress

        images = save_processed_images(processed,
                                       process_args.outpath_samples,
                                       process_args.outpath_grids,
                                       process_args.outpath_scripts,
                                       task.id,
                                       inspect=process_args.kwargs.get("need_audit", False),
                                       forbidden_review=process_args.kwargs.get("forbidden_review", False))
        images.update({'inference_time': inference_time})
        progress = TaskProgress.new_finish(task, images)
        progress.update_seed(processed.all_seeds, processed.all_subseeds)

        yield progress

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        minor_type = Txt2ImgMinorTaskType(task.minor_type)
        if minor_type <= Txt2ImgMinorTaskType.Txt2Img:
            yield from self._exec_txt2img(task)
        elif minor_type == Txt2ImgMinorTaskType.RunControlnetAnnotator:
            yield from exec_control_net_annotator(task)
