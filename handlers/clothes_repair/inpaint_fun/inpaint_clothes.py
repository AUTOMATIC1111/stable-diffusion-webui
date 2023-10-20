from PIL import Image,ImageChops
import modules.images
from modules import shared, sd_samplers, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
import modules.scripts as scripts
import numpy as np
import gradio as gr
from modules import script_callbacks
import numpy as np
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
from omegaconf import OmegaConf
from os import mkdir
from urllib import request
from modules.scripts import list_scripts
from collections import namedtuple
from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing
from enum import Enum
from typing import List, Any, Optional, Union, Tuple, Dict
import numpy as np
from modules import scripts, processing, shared, sd_models, sd_vae
from modules.api import api
import random
import os
import stat
from collections import OrderedDict
from modules.paths import models_path


path2 = os.path.abspath('.')
path2 += '/extensions/sd-webui-controlnet'
import sys

sys.path.append(path2)
# from scripts import global_state, hook, external_code, processor, batch_hijack, controlnet_version, utils

def traverse_all_files(curr_path, model_list):
    f_list = [
        (os.path.join(curr_path, entry.name), entry.stat())
        for entry in os.scandir(curr_path)
        if os.path.isdir(curr_path)
    ]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename
    return res

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors"]
cn_models_dir = os.path.join(models_path, "ControlNet")
cn_models_dir_old = os.path.join(scripts.basedir(), "models")
cn_models = OrderedDict()
cn_models_names = {}

cn_models.clear()
ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
            if extra_lora_path is not None and os.path.exists(extra_lora_path))
paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]

for path in paths:
    sort_by = shared.opts.data.get(
        "control_net_models_sort_models_by", "name")
    filter_by = shared.opts.data.get("control_net_models_name_filter", "")
    found = get_all_models(sort_by, filter_by, path)
    cn_models.update({**found, **cn_models})

# insert "None" at the beginning of `cn_models` in-place
cn_models_copy = OrderedDict(cn_models)
cn_models.clear()
cn_models.update({**{"None": None}, **cn_models_copy})

cn_models_names.clear()
for name_and_hash, filename in cn_models.items():
    if filename is None:
        continue
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    cn_models_names[name] = name_and_hash

# 模型
MODELS = list(cn_models.keys())
import copy


class InputMode(Enum):
    SIMPLE = "simple"
    BATCH = "batch"


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]


class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
            self,
            enabled: bool = True,
            module: Optional[str] = None,
            model: Optional[str] = None,
            weight: float = 1.0,
            image: Optional[InputImage] = None,
            resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT,
            low_vram: bool = False,
            processor_res: int = -1,
            threshold_a: float = 1,
            threshold_b: float = -1,
            guidance_start: float = 0.0,
            guidance_end: float = 1.0,
            pixel_perfect: bool = False,
            control_mode: Union[ControlMode, int, str] = ControlMode.CONTROL,
            **_kwargs,
    ):
        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect
        self.control_mode = control_mode

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)


class UiControlNetUnit(ControlNetUnit):
    """The data class that stores all states of a ControlNetUnit."""

    def __init__(
            self,
            input_mode: InputMode = InputMode.SIMPLE,
            batch_images: Optional[Union[str, List[InputImage]]] = None,
            output_dir: str = "",
            loopback: bool = False,
            use_preview_as_input: bool = False,
            generated_image: Optional[np.ndarray] = None,
            enabled: bool = True,
            module: Optional[str] = None,
            model: Optional[str] = None,
            weight: float = 1.0,
            image: Optional[np.ndarray] = None,
            *args,
            **kwargs,
    ):
        if use_preview_as_input and generated_image is not None:
            input_image = generated_image
            module = "none"
        else:
            input_image = image

        super().__init__(enabled, module, model, weight, input_image, *args, **kwargs)
        self.is_ui = True
        self.input_mode = input_mode
        self.batch_images = batch_images
        self.output_dir = output_dir
        self.loopback = loopback


ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


# 加载模型文件
def add_conrolnet():
    tmp_obj = UiControlNetUnit()
    tmp_obj.input_mode = InputMode.SIMPLE
    tmp_obj.batch_images = ['']
    tmp_obj.module = 'none'
    tmp_obj.weight = 1  # 权重保持在[0.5,1]之间
    tmp_obj.resize_mode = 'Crop and Resize'
    tmp_obj.low_vram = False
    tmp_obj.processor_res = -1
    tmp_obj.threshold_a = 1    # -1
    tmp_obj.threshold_b = -1
    tmp_obj.guidance_start = 0
    tmp_obj.guidance_end = 1
    tmp_obj.pixel_perfect = False
    tmp_obj.control_mode = 'Balanced'
    tmp_obj.is_ui = True
    tmp_obj.output_dir = ''
    tmp_obj.loopback = False
    tmp_obj.enabled = False
    return tmp_obj


def set_conrolnet(control_list, init_image, clo_image):

    image_array = np.array(init_image["image"])
    init_image["image"] = np.array(init_image["image"])
    clo_image_array = np.array(clo_image)
    control_list[0].enabled=True
    control_list[0].image= clo_image_array
    control_list[0].module='depth_zoe'
    control_list[0].model='control_v11f1p_sd15_depth [cfd03158]'
    # control_list[0].low_vram=True
    control_list[0].weight=0.8
    control_list[0].processor_res=clo_image.width

    control_list[1].enabled=True
    control_list[1].image = clo_image_array
    control_list[1].module='canny'
    control_list[1].model='control_v11p_sd15_canny [d14c016b]'
    # control_list[1].low_vram=True
    control_list[1].threshold_a = 150
    control_list[1].threshold_b = 200
    control_list[1].weight=0.4
    control_list[1].processor_res=clo_image.width

    control_list[2].enabled = True
    control_list[2].image = init_image
    control_list[2].module = 'inpaint_only'
    control_list[2].model = 'control_v11p_sd15_inpaint [ebff9138]'
    # control_list[2].low_vram=True
    control_list[2].weight = 1



def inpaint_clothes(init_image, clo_image, text_prompt, n_iter, *args):
    # if not init_image:
    #     raise Exception('Error！ Please add a image file!')
    if text_prompt == None:
        text_prompt = ""
    prompt = text_prompt
    negative_prompt = ""
    id_task = "task"
    steps, sampler_index = 20, 0
    restore_faces = False
    tiling = False
    denoising_strength = 0.85
    cfg_scale = 7
    batch_size = 1
    n_iter = n_iter
    width, height = init_image["image"].width, init_image["image"].height
    # width, height = 512, 512  # 生成的尺寸是否保持与原图一致
    seed = -1.0
    subseed = -1.0
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_enable_extras = False  # True,subseed到seed_resize_from_w之间的都要填值

    if init_image["image"].mode != "RGB":
        # 将图像转换为RGB模式
        init_image["image"] = init_image["image"].convert("RGB")
    else:
        init_image["image"] = init_image["image"]

    cloth_mask = init_image["mask"]
    cloth_image = init_image["image"]

    
    init_image["mask"] = np.array(init_image["mask"])[:,:,np.newaxis]
    init_image["mask"] = np.where(init_image["mask"], 255, 0)

    # print("shared.opts.sd_model_checkpoint:", shared.opts.sd_model_checkpoint)
    # print(sd_samplers.samplers_for_img2img[sampler_index].name)
    # # 加载模型和vae
    # checkpoint_path = "models/Stable-diffusion/realisticVisionV20_v20.safetensors"
    # vae_path = "models/VAE/Anything-V3.0.vae.pt"
    # model_checkpoint = sd_models.CheckpointInfo(checkpoint_path)

    # modules.sd_models.reload_model_weights(info=model_checkpoint)
    # modules.sd_vae.reload_vae_weights(vae_file = vae_path)

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[cloth_image],
        mask=None,
        mask_blur=4,
        inpainting_fill=1,
        resize_mode=0,
        denoising_strength=denoising_strength,
        image_cfg_scale=cfg_scale,
        inpaint_full_res=0,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        override_settings=[],
    )

    
    p.scripts = modules.scripts.scripts_txt2img

    tmp_scripts = []
    args_tmp = [0]
    ori_scripts = []
    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    init_from, init_to = 0, 0
    # 拿到controlnet的参数
    for script in p.scripts.alwayson_scripts:
        ori_scripts.append(script)
        if 'controlnet.py' in script.filename:
            # init_script=copy.deepcopy(script)
            init_from, init_to = script.args_from, script.args_to
            script.args_to = script.args_to - script.args_from + 1
            script.args_from = 1
            tmp_scripts.append(script)

    # 只保留controlnet组件
    p.scripts.alwayson_scripts = tmp_scripts

    # 改变controlnet对象的值
    for script in p.scripts.alwayson_scripts:
        # max_models = shared.opts.data.get("control_net_max_models_num", 1)
        max_models = 3
        control_list = []
        for i in range(max_models):
            control_list.append(add_conrolnet())
        print("controlnetlist:", len(control_list))
        set_conrolnet(control_list, init_image, clo_image)
        args_tmp[script.args_from:script.args_to] = control_list
    print(args_tmp, file=sys.stderr)
    p.script_args = args_tmp
    p.extra_generation_params["Mask blur"] = 4
    processed = modules.scripts.scripts_img2img.run(p, *args_tmp)
    if processed is None:
        print("-------------------------------------------------------------")
        processed = process_images(p)
    

    # 还原回去
    for script in p.scripts.alwayson_scripts:
        if 'controlnet.py' in script.filename:
            script.args_from, script.args_to = init_from, init_to
    p.scripts.alwayson_scripts = ori_scripts
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []
    # processed.images = [processed.images[0]]
    # return processed.images
    return processed,p

