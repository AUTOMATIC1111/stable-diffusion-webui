#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 12:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : img2img.py
# @Software: Hifive
import os.path
import time
import typing
import modules.shared as shared
import numpy as np
from enum import IntEnum
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
from modules.scripts import scripts_img2img
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.img2img import process_batch
from worker.task import TaskHandler, TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images
from extension.controlnet import exec_control_net_annotator
from handlers.dumper import TaskDumper
from modules import sd_models

AlwaysonScriptsType = typing.Mapping[str, typing.Mapping[str, typing.Any]]


class Img2ImgMinorTaskType(IntEnum):
    Img2Img = 0
    RunControlnetAnnotator = 100


class Img2ImgTask(StableDiffusionProcessingImg2Img):

    # {"all_negative_prompts": null, "all_prompts": null, "all_seeds": null, "all_subseeds": null, "batch_size": 1,
    # "cfg_scale": 7, "color_corrections": null, "compress_pnginfo": false, "ddim_discretize": "uniform",
    # "denoising_strength": 0.75, "disable_extra_networks": false, "do_not_reload_embeddings": false,
    # "do_not_save_grid": false, "do_not_save_samples": false, "eta": null, "extra_generation_params": {},
    # "height": 512, "image_cfg_scale": null, "image_conditioning": null,
    # "image_mask": "image_mask.png", "init_images": ["init_images_0.png"], "init_latent": null, "initial_noise_multiplier": 1.0, "inpaint_full_res": 0,
    # "inpaint_full_res_padding": 32, "inpainting_fill": 1, "inpainting_mask_invert": 0,
    # "is_using_inpainting_conditioning": false, "iteration": 0, "latent_mask": null,
    # "mask": null, "mask_blur": 4, "mask_for_overlay": null, "n_iter": 1, "negative_prompt": "", "nmask": null, "outpath_grids": "outputs/img2img-grids",
    # "outpath_samples": "outputs/img2img-images", "overlay_images": null, "override_settings": {},
    # "override_settings_restore_afterwards": true, "paste_to": null, "prompt": "", "prompt_for_display": null, "resize_mode": 0, "restore_faces": false,
    # "s_churn": 0.0, "s_noise": 1.0, "s_tmax": Infinity, "s_tmin": 0.0, "sampler": null, "sampler_name": "Euler a",
    # "sampler_noise_scheduler_override": null, "script_args": null, "scripts": null, "seed": -1.0, "seed_resize_from_h": 0,
    # "seed_resize_from_w": 0, "steps": 20, "styles": [], "subseed": -1, "subseed_strength": 0, "tiling": false, "width": 512}

    def __init__(self, base_model_path: str,
                 user_id: str,
                 default_script_arg_img2img: typing.Sequence,  # 默认脚本参数，handler构造。
                 prompt: str,  # TAG
                 negative_prompt: str,  # 反向TAG
                 sampler_name: str = None,  # 采样器
                 init_img: str = None,  # 原图片（i2i，MODE=0）
                 sketch: str = None,  # 绘图图片（绘图，MODE=1）
                 init_img_with_mask: typing.Mapping[str, str] = None,
                 # 局部重绘图（手涂蒙版，MODE=2）,字典形式必须包含image,mask 2个KEY,蒙版为画笔白色其他地方为黑色
                 init_img_inpaint: str = None,  # 局部重绘图（上传蒙版，MODE=4）
                 init_mask_inpaint: str = None,  # 局部重绘蒙版图（上传蒙版，MODE=4）
                 inpaint_color_sketch: str = None,  # 局部重新绘制图片手绘（局部重新绘制手涂，MODE=3）
                 inpaint_color_sketch_orig: str = None,  # 局部重新绘制图片原图（局部重新绘制手涂，MODE=3）
                 batch_size: int = 1,  # 批次
                 mask_blur: int = 4,  # 蒙版模糊
                 n_iter: int = 1,  # 迭代数
                 steps: int = 30,  # 步数
                 cfg_scale: float = 7.0,  # 提示词相关性
                 image_cfg_scale: float = 1.5,  # 图片相关性，一般是隐藏的不会使用
                 width: int = 512,  # 图宽
                 height: int = 512,  # 图高
                 restore_faces: bool = False,  # 面部修复
                 tiling: bool = False,  # 可平铺
                 mode: int = 0,  # i2i 模式
                 seed: int = -1,  # 随机种子
                 seed_enable_extras: bool = False,  # 是否启用随机种子扩展
                 subseed: int = -1,  # 差异随机种子
                 subseed_strength: float = 0,  # 差异强度
                 seed_resize_from_h: int = 0,  # 重置尺寸种子-高度
                 seed_resize_from_w: int = 0,  # 重置尺寸种子-宽度
                 resize_mode: int = 0,  # 局部重绘（含手涂和上传）的缩放模式，0-拉伸，1-裁剪，2-填充
                 inpainting_mask_invert: int = 0,  # 局部重绘（含手涂和上传）的蒙版模式，0-重绘蒙版内容，1-重绘非蒙版内容
                 inpaint_full_res: bool = False,  # 局部重绘（含手涂和上传）的重绘区域，默认原图；True代表仅蒙版
                 inpaint_full_res_padding: int = 32,  # 局部重绘（含手涂和上传）的仅蒙版模式的边缘预留像素
                 inpainting_fill: int = 1,  # 局部重绘（含手涂和上传）的蒙版蒙住的内容：0-填充，1-原图，2-潜变量，3-潜变量为0
                 mask_alpha: float = 0,  # 局部重绘制（含手涂和上传），蒙版透明度
                 denoising_strength: float = 0.75,  # 重绘幅度（含手涂和上传）
                 select_script_name: str = None,  # 选择下拉框脚本名
                 select_script_args: typing.Sequence = None,  # 选择下拉框脚本参数
                 alwayson_scripts: AlwaysonScriptsType = None,  # 插件脚本，object格式： {插件名: {'args': [参数列表]}}
                 img2img_batch_input_dir: str = None,
                 img2img_batch_output_dir: str = None,
                 prompt_styles: typing.List[str] = None,  # 提示风格（模板风格也就是TAG模板）
                 img2img_batch_inpaint_mask_dir: str = None,
                 compress_pnginfo: bool = True,  # 使用GZIP压缩图片信息（默认开启）
                 override_settings_texts=None,  # 自定义设置 TEXT,如: ['Clip skip: 2', 'ENSD: 31337']
                 lora_models: typing.Sequence[str] = None,  # 使用LORA，用户和系统全部LORA列表
                 embendings: typing.Sequence[str] = None,  # 使用embending，用户和系统全部mbending列表
                 **kwargs):
        override_settings = create_override_settings_dict(override_settings_texts or [])
        image = None
        mask = None
        self.is_batch = False
        if mode == 5:
            if not img2img_batch_input_dir \
                    or not img2img_batch_output_dir:
                raise ValueError('batch input or output directory is empty')
            self.is_batch = True
        elif mode == 4:
            if not init_img_inpaint or not init_mask_inpaint:
                raise ValueError('img_inpaint or mask_inpaint not found')
            image = Image.open(init_img_inpaint)
            mask = Image.open(init_mask_inpaint)
        elif mode == 3:
            if not inpaint_color_sketch:
                raise Exception('inpaint_color_sketch not found')
            image = Image.open(inpaint_color_sketch)

            orig_path = inpaint_color_sketch_orig or inpaint_color_sketch
            if orig_path != inpaint_color_sketch:
                orig = Image.open(orig_path)
            else:
                orig = image
            pred = np.any(np.array(image) != np.array(orig), axis=-1)
            mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
            mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
            blur = ImageFilter.GaussianBlur(mask_blur)
            image = Image.composite(image.filter(blur), orig, mask.filter(blur))
            image = image.convert("RGB")
        elif mode == 2:
            if not init_img_with_mask:
                raise Exception('init_img_with_mask not found')
            if 'mask' not in init_img_with_mask or 'image' not in init_img_with_mask:
                raise Exception('mask or image not found in init_img_with_mask')
            image_path, mask_path = init_img_with_mask["image"], init_img_with_mask["mask"]
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert("RGB")
        elif mode == 1:
            if not sketch:
                raise Exception('sketch not found')
            image = Image.open(sketch).convert("RGB")
            mask = None
        elif mode == 0:
            if not init_img:
                raise Exception('init image not found')
            image = Image.open(init_img).convert("RGB")
            mask = None
        else:
            raise ValueError(f'mode value error, except 0~5 got {mode}')
        if image is not None:
            image = ImageOps.exif_transpose(image)

        assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

        if not scripts_img2img:
            scripts_img2img.initialize_scripts(True)

        i2i_script_runner = scripts_img2img
        selectable_scripts, selectable_script_idx = get_selectable_script(i2i_script_runner, select_script_name)
        script_args = init_script_args(default_script_arg_img2img, alwayson_scripts, selectable_scripts,
                                       selectable_script_idx, select_script_args, i2i_script_runner)

        super(Img2ImgTask, self).__init__(
            sd_model=shared.sd_model,
            outpath_samples=f"output/img2img/{user_id}/samples/",
            outpath_grids=f"output/img2img/{user_id}/grids/",
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=prompt_styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            seed_enable_extras=seed_enable_extras,
            sampler_name=sampler_name or 'Euler a',
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=tiling,
            init_images=[image],
            mask=mask,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            resize_mode=resize_mode,
            denoising_strength=denoising_strength,
            image_cfg_scale=image_cfg_scale,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            override_settings=override_settings,
        )
        self.scripts = i2i_script_runner
        self.script_name = select_script_name
        self.base_model_path = base_model_path
        self.selectable_scripts = selectable_scripts
        self.img2img_batch_input_dir = img2img_batch_input_dir
        self.img2img_batch_output_dir = img2img_batch_output_dir
        self.img2img_batch_inpaint_mask_dir = img2img_batch_inpaint_mask_dir
        self.extra_generation_params["Mask blur"] = mask_blur
        self.compress_pnginfo = compress_pnginfo
        self.kwargs = kwargs
        self.loras = lora_models
        self.embending = embendings

        if selectable_scripts:
            self.script_args = script_args
        else:
            self.script_args = tuple(script_args)

    @classmethod
    def from_task(cls, task: Task, default_script_arg_img2img: typing.Sequence):
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
        kwargs = task.data.copy()
        kwargs.pop('base_model_path')
        kwargs.pop('alwayson_scripts')
        kwargs.pop('prompt')
        kwargs.pop('negative_prompt')
        kwargs.pop('user_id')
        kwargs.pop('select_script')

        return cls(base_model_path,
                   user_id,
                   default_script_arg_img2img,
                   prompt=prompt,
                   negative_prompt=negative_prompt,
                   alwayson_scripts=alwayson_scripts,
                   select_script_name=select_script_name,
                   select_script_args=select_script_args,
                   **kwargs)

    @staticmethod
    def debug_task() -> typing.Sequence[Task]:
        models_dir = '/home/jty/stable-diffusion-webui-master/models/Stable-diffusion'
        models = [
            'v1-5-pruned-emaonly.ckpt',
            'chilloutmix_NiCkpt.ckpt',
            'guofeng2_v20.safetensors',
            'v1-5-pruned-emaonly.ckpt',
        ]
        model_hash_map = {
            'v1-5-pruned-emaonly.ckpt': 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516',
            'chilloutmix_NiCkpt.ckpt': '3a17d0deffa4592fd91c711a798031a258ab44041809ade8b4591c0225ea9401',
            'guofeng2_v20.safetensors': '3257896d4b399dc70cd0d2ef76f4965d309413fea1f11f1d3173e9069e3b3a92'
        }

        t = {
            'task_id': 'test_i2i',
            'base_model_path': f'{models_dir}/v1-5-pruned-emaonly.ckpt',
            'model_hash': '',
            'alwayson_scripts': {},
            'user_id': 'test_user',
            'select_script': None,
            'task_type': TaskType.Image2Image,
            'create_at': -1,
            'negative_prompt': '',
            "init_img": "test-imgs/QQ20230324-104509.png",
            'prompt': '<lora:Xiaorenshu_v20:0.6>',
            'lora_models': ['/data/apksamba/sd/models/Lora/Xiaorenshu_v20.safetensors']
        }

        tasks = []
        for m in models:
            basename, _ = os.path.splitext(m)
            t['base_model_path'] = os.path.join(models_dir, m)
            t['task_id'] = f'test_i2i_{basename}_{len(tasks)}'
            t['model_hash'] = model_hash_map[m]
            tasks.append(Task(**t))
        return tasks


class Img2ImgTaskHandler(TaskHandler):

    def __init__(self):
        super(Img2ImgTaskHandler, self).__init__(TaskType.Image2Image)
        self._default_script_args_load_t = 0

    def _refresh_default_script_args(self):
        if time.time() - self._default_script_args_load_t > 3600 * 4:
            self._load_default_script_args()

    def _load_default_script_args(self):
        self.default_script_args = init_default_script_args(scripts_img2img)
        self._default_script_args_load_t = time.time()

    def _build_img2img_arg(self, task: Task) -> Img2ImgTask:
        # 可不使用定时刷新，直接初始化。
        self._refresh_default_script_args()

        return Img2ImgTask.from_task(task, self.default_script_args)

    def _exec_img2img(self, task: Task) -> typing.Iterable[TaskProgress]:
        base_model_path = task.sd_model_path
        progress = TaskProgress.new_ready(task, 'at the ready')
        load_sd_model_weights(base_model_path)
        progress.task_desc = f'model loaded:{os.path.basename(base_model_path)}, run i2i...'
        progress.status = TaskStatus.Ready
        yield progress
        # 参数有使用到sd_model因此在切换模型后再构造参数。
        process_args = self._build_img2img_arg(task)
        if process_args.loras:
            # 设置LORA，具体实施在modules/exta_networks.py 中activate函数。
            sd_models.user_loras = [f for f in process_args.loras if os.path.isfile(f)]
        else:
            sd_models.user_loras = []
        if process_args.embending:
            embending_dirs = [
                os.path.dirname(p) for p in process_args.embending if os.path.isfile(p)
            ]
            sd_models.user_embending_dirs = set(embending_dirs)
        else:
            sd_models.user_embending_dirs = []

        progress.status = TaskStatus.Running
        progress.task_desc = f'i2i task({task.id}) running'
        yield progress
        if process_args.is_batch:
            assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

            process_batch(process_args,
                          process_args.img2img_batch_input_dir,
                          process_args.img2img_batch_output_dir,
                          process_args.img2img_batch_inpaint_mask_dir,
                          process_args.script_args)

            processed = Processed(process_args, [], process_args.seed, "")
        else:
            if process_args.selectable_scripts:
                processed = process_args.scripts.run(process_args,
                                                     *process_args.script_args)  # Need to pass args as list here
            else:
                processed = process_images(process_args)

        process_args.close()
        images = save_processed_images(processed, process_args.outpath_samples, task.id)
        progress.set_finish_result(images)
        yield progress

    def _set_task_status(self, p: TaskProgress):
        super()._set_task_status(p)
        dumper = TaskDumper()
        dumper.dump_task_progress(p)

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        minor_type = Img2ImgMinorTaskType(task.minor_type)
        if minor_type == Img2ImgMinorTaskType.Img2Img:
            yield from self._exec_img2img(task)
        elif minor_type == Img2ImgMinorTaskType.RunControlnetAnnotator:
            yield from exec_control_net_annotator(task)
