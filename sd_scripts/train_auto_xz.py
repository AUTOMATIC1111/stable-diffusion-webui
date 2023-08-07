import os #,sys,re,torch
from PIL import Image, ImageOps
# import random
import time
import tqdm
from enum import Enum
import math
# import diffusers
# from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
# from diffusers import (
#     StableDiffusionPipeline,
#     DDPMScheduler,
#     EulerAncestralDiscreteScheduler,
#     DPMSolverMultistepScheduler,
#     DPMSolverSinglestepScheduler,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     DDIMScheduler,
#     EulerDiscreteScheduler,
#     HeunDiscreteScheduler,
#     KDPM2DiscreteScheduler,
#     KDPM2AncestralDiscreteScheduler,
# )
from accelerate import Accelerator
# from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from .train_network_all_auto import train_with_params, train_callback

# from prompts_generate import txt2img_prompts
from sd_scripts.finetune.tag_images_by_wd14_tagger import tagger
from sd_scripts.library import autocrop
from sd_scripts.super_upscaler.super_upscaler import upscaler

# from 

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"
# MODEL_PATH = "/data/qll/stable-diffusion-webui/models"

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def resize_image(resize_mode, im, width, height, upscaler_name=None, models_path=""):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    upscaler_name = upscaler_name

    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            # upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            # if len(upscalers) == 0:
            #     upscaler = shared.sd_upscalers[0]
            #     print(f"could not find upscaler named {upscaler_name or '<empty string>'}, using {upscaler.name} as a fallback")
            # else:
            #     upscaler = upscalers[0]

            # im = upscaler.scaler.upscale(im, scale, upscaler.data_path)
            im = upscaler(im, upscale_by=scale, style_type=1, upscaler_2_visibility=0.3, swap=True,models_path=models_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res

def save_pic(image, index, params, existing_caption):
    # save_pic_with_caption(image, index, params, existing_caption=existing_caption)
    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image.save(os.path.join(params.dstdir, f"{basename}.png"))

    if params.flip:
        # save_pic_with_caption(ImageOps.mirror(image), index, params, existing_caption=existing_caption)
        image = ImageOps.mirror(image)
        image.save(os.path.join(params.dstdir, f"{basename}.png"))


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted


# not using torchvision.transforms.CenterCrop because it doesn't allow float regions
def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    return wh and center_crop(image, *wh)

class PreprocessTxtAction(Enum):
    Prepend = 'prepend'
    Append = 'append'
    Copy = 'copy'
    Ignore = 'ignore'

class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    process_caption = False
    process_caption_deepbooru = False
    preprocess_txt_action = None

    def do_caption(self, existing_caption=None):
        '''
        忽略已存在TAG
        '''
        if not existing_caption:
            return True
        return self.preprocess_txt_action in [
            PreprocessTxtAction.Prepend, PreprocessTxtAction.Copy, PreprocessTxtAction.Append
        ]

# 一键图片预处理函数
def train_preprocess(process_src, process_dst, process_width, process_height, preprocess_txt_action, process_keep_original_size,
                    process_flip, process_split, process_caption, process_caption_deepbooru=False, split_threshold=0.5,
                    overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
                    process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                    process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                    process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                    process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None, model_path=""):
    width = process_width
    height = process_height
    src = os.path.abspath(process_src)
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    # assert src != dst, 'same directory specified as source and destination'

    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)  # listfiles(src)

    # shared.state.job = "preprocess"
    # shared.state.textinfo = "Preprocessing..."
    # shared.state.job_count = len(files)

    params = PreprocessParams()
    params.dstdir = dst
    params.flip = process_flip
    params.process_caption = process_caption
    params.process_caption_deepbooru = process_caption_deepbooru
    params.preprocess_txt_action = preprocess_txt_action

    pbar = tqdm.tqdm(files)
    for index, imagefile in enumerate(pbar):
        # params.subindex = 0
        filename = os.path.join(src, imagefile)
        try:
            img = Image.open(filename)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception:
            continue

        description = f"Preprocessing [Image {index}/{len(files)}]"
        pbar.set_description(description)
        # shared.state.textinfo = description

        params.src = filename

        existing_caption = None
        existing_caption_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(existing_caption_filename):
            with open(existing_caption_filename, 'r', encoding="utf8") as file:
                existing_caption = file.read()

        # if shared.state.interrupted:
        #     break

        if img.height > img.width:
            ratio = (img.width * height) / (img.height * width)
            inverse_xy = False
        else:
            ratio = (img.height * width) / (img.width * height)
            inverse_xy = True

        process_default_resize = True

        if process_split and ratio < 1.0 and ratio <= split_threshold:
            for splitted in split_pic(img, inverse_xy, width, height, overlap_ratio):
                save_pic(splitted, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                dnn_model_path = autocrop.download_and_cache_models(os.path.join(model_path, "opencv"))
            except Exception as e:
                print(
                    "Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.",
                    e)

            autocrop_settings = autocrop.Settings(
                crop_width=width,
                crop_height=height,
                face_points_weight=process_focal_crop_face_weight,
                entropy_points_weight=process_focal_crop_entropy_weight,
                corner_points_weight=process_focal_crop_edges_weight,
                annotate_image=process_focal_crop_debug,
                dnn_model_path=dnn_model_path,
            )
            for focal in autocrop.crop_image(img, autocrop_settings):
                save_pic(focal, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_multicrop:
            cropped = multicrop_pic(img, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea,
                                    process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)
            if cropped is not None:
                save_pic(cropped, index, params, existing_caption=existing_caption)
            else:
                print(
                    f"skipped {img.width}x{img.height} image {filename} (can't find suitable size within error threshold)")
            process_default_resize = False

        if process_keep_original_size:
            if img.width>3096 or img.height>3096:  # 对于超大图片限制在2048的范围内
                if img.width>img.height:
                    ratio = 2048.0/img.width
                else:
                    ratio = 2048.0/img.height
                img = resize_image(1, img, int(img.width*ratio), int(img.height*ratio))
            save_pic(img, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_default_resize:
            img = resize_image(1, img, width, height)
            save_pic(img, index, params, existing_caption=existing_caption)

        # shared.state.nextjob()
        if callable(progress_cb):
            current_progress = (index + 1) * 100 / len(files)
            if current_progress % 5 == 0:
                progress_cb(current_progress)
    return 

# 一键打标函数
def train_tagger(train_data_dir,model_dir,trigger_word=None,undesired_tags=None,general_threshold=0.35,character_threshold=0.35):
    tagger(
        train_data_dir=train_data_dir,
        model_dir= model_dir, #r"/data/qll/stable-diffusion-webui/models/tag_models",
        force_download=False,
        batch_size=8,
        max_data_loader_n_workers=4,
        caption_extension=".txt",
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        recursive=True,
        remove_underscore=True,
        undesired_tags=undesired_tags,
        frequency_tags=False,
        addtional_tags=trigger_word
    )   
    return


MODEL_PATH = "/data/qll/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors"
PIC_SAVE_PATH = "/data/qll/pics/yijian_sanciyuan_train"
LORA_PATH = "/data/qll/stable-diffusion-webui/models/Lora"

def prepare_accelerator(logging_dir="./logs",log_prefix=None,gradient_accumulation_steps=1,mixed_precision="no"):
    if logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if log_prefix is None else log_prefix
        logging_dir = logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=None,
        project_dir=logging_dir,
    )

    # accelerateの互換性問題を解決する
    accelerator_0_15 = True
    try:
        accelerator.unwrap_model("dummy", True)
        print("Using accelerator 0.15.0 or above.")
    except TypeError:
        accelerator_0_15 = False

    def unwrap_model(model):
        if accelerator_0_15:
            return accelerator.unwrap_model(model, True)
        return accelerator.unwrap_model(model)

    return accelerator, unwrap_model

from multiprocessing import Process


class Train_Params:
    def __init__(self,
        pretrained_model_name_or_path=None,
        network_weights=None,
        output_name = "",
        output_dir = "",
        save_every_n_epochs=None,
        save_last_n_epochs = 1,
        trigger_words = [""],
        list_train_data_dir=[],
        num_repeats=["50"],
        batch_size=8,
        resolution="512,768",
        epoch=10, 
        network_module="networks.lora",
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
        network_dim=32,
        network_alpha=32,
        clip_skip=2,
        enable_preview=False,
        sample_prompts="",
        sample_sampler="euler_a",
        optimizer_type="AdamW8bit",
        learning_rate=0.0001,
        unet_lr=0.0001,
        text_encoder_lr=0.00001,
        lr_scheduler="cosine_with_restarts",
        auto_lr=False,
        auto_lr_param=1,
        cache_latents=True,
        cache_latents_to_disk=False,
        enable_bucket=True,
        min_bucket_reso=256,
        max_bucket_reso=2048,
        bucket_reso_steps=64,
        bucket_no_upscale=True,
        token_warmup_min=1,
        token_warmup_step=0,

        caption_extension=".txt",
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0, 
        caption_tag_dropout_rate=0.0, 
        shuffle_caption=False, 
        weighted_captions=False, 
        keep_tokens=0,
        color_aug=False, 
        flip_aug=False, 
        face_crop_aug_range=None,
        random_crop=False,

        lowram=False,
        mem_eff_attn=False,
        xformers=True,
        vae=None, 
        max_data_loader_n_workers=8,
        persistent_data_loader_workers=True,

        max_train_steps=1600, 
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        full_fp16=True, 

        sample_every_n_epochs=None,

        conv_dim=None, 
        conv_alpha=None,
        unit=8,
        dropout=0,
        algo='lora', 

        enable_block_weights=False,
        block_dims=None,
        block_alphas=None, 
        conv_block_dims=None,
        conv_block_alphas=None, 
        down_lr_weight=None,
        mid_lr_weight=None,
        up_lr_weight=None,
        block_lr_zero_threshold=0.0,
        weight_decay=0.01,
        betas=0.9,

        max_grad_norm=1.0, 

        lr_scheduler_num_cycles=1, 
        lr_warmup_steps=0,
        lr_scheduler_power=1,
        seed=918699631,
        prior_loss_weight=1.0, 
        min_snr_gamma=None, 
        noise_offset=None, 
        adaptive_noise_scale=None,  
        multires_noise_iterations=6, 
        multires_noise_discount=0.3, 

        config_file=None,
        output_config=False,
        ) -> None:
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.network_weights=network_weights
        self.output_name = output_name
        self.output_dir = output_dir
        self.save_every_n_epochs=save_every_n_epochs
        self.save_last_n_epochs = save_last_n_epochs
        self.trigger_words = trigger_words
        self.list_train_data_dir=list_train_data_dir
        self.num_repeats=num_repeats
        self.batch_size=batch_size
        self.resolution=resolution
        self.epoch=epoch
        self.network_module=network_module
        self.network_train_unet_only=network_train_unet_only
        self.network_train_text_encoder_only=network_train_text_encoder_only
        self.network_dim=network_dim
        self.network_alpha=network_alpha
        self.clip_skip=clip_skip
        self.enable_preview=enable_preview
        self.sample_prompts=sample_prompts
        self.sample_sampler=sample_sampler
        self.optimizer_type=optimizer_type
        self.learning_rate=learning_rate
        self.unet_lr=unet_lr
        self.text_encoder_lr=text_encoder_lr
        self.lr_scheduler=lr_scheduler
        self.auto_lr=auto_lr
        self.auto_lr_param=auto_lr_param
        self.cache_latents=cache_latents
        self.cache_latents_to_disk=cache_latents_to_disk
        self.enable_bucket=enable_bucket
        self.min_bucket_reso=min_bucket_reso
        self.max_bucket_reso=max_bucket_reso
        self.bucket_reso_steps=bucket_reso_steps
        self.bucket_no_upscale=bucket_no_upscale
        self.token_warmup_min=token_warmup_min
        self.token_warmup_step=token_warmup_step

        self.caption_extension=caption_extension
        self.caption_dropout_rate=caption_dropout_rate
        self.caption_dropout_every_n_epochs=caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate=caption_tag_dropout_rate 
        self.shuffle_caption=shuffle_caption
        self.weighted_captions=weighted_captions
        self.keep_tokens=keep_tokens
        self.color_aug=color_aug
        self.flip_aug=flip_aug
        self.face_crop_aug_range=face_crop_aug_range
        self.random_crop=random_crop

        self.lowram=lowram
        self.mem_eff_attn=mem_eff_attn
        self.xformers=xformers
        self.vae=vae
        self.max_data_loader_n_workers=max_data_loader_n_workers
        self.persistent_data_loader_workers=persistent_data_loader_workers

        self.max_train_steps=max_train_steps
        self.gradient_checkpointing=gradient_checkpointing
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.mixed_precision=mixed_precision
        self.full_fp16=full_fp16

        self.sample_every_n_epochs=sample_every_n_epochs

        self.conv_dim=conv_dim 
        self.conv_alpha=conv_alpha
        self.unit=unit
        self.dropout=dropout
        self.algo=algo

        self.enable_block_weights=enable_block_weights
        self.block_dims=block_dims
        self.block_alphas=block_alphas
        self.conv_block_dims=conv_block_dims
        self.conv_block_alphas=conv_block_alphas
        self.down_lr_weight=down_lr_weight
        self.mid_lr_weight=mid_lr_weight
        self.up_lr_weight=up_lr_weight
        self.block_lr_zero_threshold=block_lr_zero_threshold
        self.weight_decay=weight_decay
        self.betas=betas

        self.max_grad_norm=max_grad_norm

        self.lr_scheduler_num_cycles=lr_scheduler_num_cycles
        self.lr_warmup_steps=lr_warmup_steps
        self.lr_scheduler_power=lr_scheduler_power
        self.seed=seed
        self.prior_loss_weight=prior_loss_weight
        self.min_snr_gamma=min_snr_gamma
        self.noise_offset=noise_offset
        self.adaptive_noise_scale=adaptive_noise_scale  
        self.multires_noise_iterations=multires_noise_iterations
        self.multires_noise_discount=multires_noise_discount

        self.config_file=config_file
        self.output_config=output_config


def train_callback(percentage):
    print(percentage)


# 一键训练入口函数
def train_auto(
    train_data_dir="",  # 训练的图片路径
    train_type=0,  # 训练的类别
    task_id=0,   # 任务id,作为Lora名称
    sd_model_path="", # 底模路径
    lora_path="", # 文件夹名字
    general_model_path="", # 通用路径,
    train_callback=None, # callback函数
    other_args=None # 预留一个，以备用
):
    # 预设参数
    width = 512
    height = 768
    trigger_word = task_id
    undesired_tags = ""  # 待测试五官

    # 1.图片预处理
    train_preprocess(process_src=train_data_dir, process_dst=train_data_dir, process_width=width, process_height=height, preprocess_txt_action='ignore', process_keep_original_size=False,
                    process_flip=False, process_split=False, process_caption=False, process_caption_deepbooru=False, split_threshold=0.5,
                    overlap_ratio=0.2, process_focal_crop=True, process_focal_crop_face_weight=0.9,
                    process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5,
                    process_focal_crop_debug=False, process_multicrop=None, process_multicrop_mindim=None,
                    process_multicrop_maxdim=None, process_multicrop_minarea=None, process_multicrop_maxarea=None,
                    process_multicrop_objective=None, process_multicrop_threshold=None, progress_cb=None, model_path=general_model_path)
    train_callback(2)
    #2.tagger反推
    tagger_path=os.path.join(general_model_path,"tag_models")
    cp = Process(target=train_tagger,args=(train_data_dir,tagger_path,trigger_word,undesired_tags,0.35,0.35)) 
    cp.start()
    cp.join()
    train_callback(5)

    lora_name = f"{task_id}"
    num_repeats = int(os.getenv("AUTO_TRAIN_REPEATS", "50"))

    # 3.自动训练出图
    train_with_params(
        pretrained_model_name_or_path=sd_model_path,
        network_weights=None,
        output_name = lora_name,#f"num_r{num_r}_epo{epo}_net{net}_lr{al}",
        output_dir = lora_path,
        save_every_n_epochs=None,
        save_last_n_epochs = 1,
        trigger_words = [task_id],#[f"{task_id}",f"{task_id}"],
        list_train_data_dir=[train_data_dir],
        num_repeats=[f"{num_repeats}"],
        batch_size=4,
        resolution=f"{width},{height}",
        epoch=10,  # 整数，随便填
        network_module="networks.lora",
        network_train_unet_only=False,
        network_train_text_encoder_only=False,
        network_dim=32,  # 4的倍数，<=256
        network_alpha=32,  # 小于等于network_dim,可以不是4的倍数
        clip_skip=2,  # 0-12
        enable_preview=False,  # 和下面这几个参数一起的
        sample_prompts="",  # 文件路径，比如c:\promts.txt,file for prompts to generate sample images
        sample_sampler="euler_a",
        optimizer_type="AdamW8bit",
        learning_rate=0.0001/2.0,
        unet_lr=0.0001/2.0,
        text_encoder_lr=0.00001/2.0,
        lr_scheduler="cosine_with_restarts",
        auto_lr=False,
        auto_lr_param=1,

        cache_latents=True,
        # cache latents to main memory to reduce VRAM usage (augmentations must be disabled)
        cache_latents_to_disk=False,
        # cache latents to disk to reduce VRAM usage (augmentations must be disabled)
        enable_bucket=True,  # enable buckets for multi aspect ratio training
        min_bucket_reso=256,  # 范围自己定，minimum resolution for buckets
        max_bucket_reso=2048,  # 范围自己定，maximum resolution for buckets
        bucket_reso_steps=64,  # 秋叶版没有这个,steps of resolution for buckets, divisible by 8 is recommended
        bucket_no_upscale=True,  # 秋叶版没有这个,make bucket for each image without upscaling
        token_warmup_min=1,  # 秋叶版没有这个,start learning at N tags (token means comma separated strinfloatgs)
        token_warmup_step=0, # 秋叶版没有这个,tag length reaches maximum on N steps (or N*max_train_steps if N<1)

        caption_extension=".txt",
        caption_dropout_rate=0.0,  # Rate out dropout caption(0.0~1.0)
        caption_dropout_every_n_epochs=0,  # Dropout all captions every N epochs
        caption_tag_dropout_rate=0.0,  # Rate out dropout comma separated tokens(0.0~1.0)
        shuffle_caption=False,  # shuffle comma-separated caption
        weighted_captions=False,  # 使用带权重的 token，不推荐与 shuffle_caption 一同开启
        keep_tokens=0,
        # keep heading N tokens when shuffling caption tokens (token means comma separated strings)
        color_aug=False,  # 秋叶版没有这个,enable weak color augmentation
        flip_aug=False,  # 秋叶版没有这个,enable horizontal flip augmentation
        face_crop_aug_range=None, # 1.0,2.0,4.0
        # 秋叶版没有这个,enable face-centered crop augmentation and its range (e.g. 2.0,4.0)
        random_crop=False,
        # 秋叶版没有这个,enable random crop (for style training in face-centered crop augmentation)

        lowram=False,
        # enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)
        mem_eff_attn=False,  # use memory efficient attention for CrossAttention
        xformers=True,  # 如果mem_eff_attn为True则xformers设置无效
        vae=None,  # 比如：c:\vae.pt, 秋叶版没有这个,path to checkpoint of vae to replace
        max_data_loader_n_workers=8,
        # 秋叶版没有这个,max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)
        persistent_data_loader_workers=True,
        # persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory)

        max_train_steps=1600,  # 秋叶版没有这个,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        # 整数，随便填, Number of updates steps to accumulate before performing a backward/update pass
        mixed_precision="no",  # 是否使用混精度
        full_fp16=True,  # fp16 training including gradients

        # ["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",]
        sample_every_n_epochs=None,  # 1,2,3,4,5.....

        # network额外参数
        conv_dim=None,  # 默认为None，可以填4的倍数，类似于network_dim,
        # lycoris才有，# 4的倍数, 适用于lora,dylora。如果是dylora,则"conv_dim must be same as network_dim",
        conv_alpha=None,  # 默认为None，可以填比con_dim小的整数，类似于network_alpha； lycoris才有，<=conv_dim, 适用于lora,dylora
        unit=8,  # 秋叶版没有
        dropout=0,  # dropout 概率, 0 为不使用 dropout, 越大则 dropout 越多，推荐 0~0.5， LoHa/LoKr/(IA)^3暂时不支持
        algo='lora',  # 可选['lora','loha','lokr','ia3']

        enable_block_weights=False,  # 让下面几个参数有效
        block_dims=None,  # lora,  类似于network_dim,
        block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
        conv_block_dims=None,  # lora,  类似于network_dim,
        conv_block_alphas=None,  # lora,默认为None，可以填比con_dim小的整数，类似于network_alpha
        down_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        mid_lr_weight=None,  # lora, 1位float,例如 1.0；
        up_lr_weight=None,  # lora, 12位的float List，例如[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        block_lr_zero_threshold=0.0,  # float型，分层学习率置 0 阈值

        
        # AdamW (default), AdamW8bit, Lion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, AdaFactor
        weight_decay=0.01,  # optimizer_args,优化器内部的参数，权重衰减系数，不建议随便改
        betas=0.9,  # optimizer_args,优化器内部的参数，不建议随便改

        max_grad_norm=1.0,  # Max gradient norm, 0 for no clipping

        # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor
        lr_scheduler_num_cycles=1,  # Number of restarts for cosine scheduler with restarts
        lr_warmup_steps=0,  # Number of steps for the warmup in the lr scheduler
        lr_scheduler_power=1,  # Polynomial power for polynomial scheduler

        seed=918699631,
        prior_loss_weight=1.0,  # loss weight for regularization images
        min_snr_gamma=None,  # float型，比如5.0，最小信噪比伽马值，如果启用推荐为 5
        noise_offset=None,  # float型，0.1左右,enable noise offset with this value (if enabled, around 0.1 is recommended)
        adaptive_noise_scale=None,  #  float型， 1.0
        # 与noise_offset配套使用；add `latent mean absolute value * this value` to noise_offset (disabled if None, default)
        multires_noise_iterations=6,  # 整数，多分辨率（金字塔）噪声迭代次数 推荐 6-10。无法与 noise_offset 一同启用。
        multires_noise_discount=0.3,  # 多分辨率（金字塔）噪声衰减率 推荐 6-10。无法与 noise_offset 一同启用。

        config_file=None,#"test_config.toml",  # using .toml instead of args to pass hyperparameter
        output_config=False,  # output command line args to given .toml file
        # accelerator=accelerator,
        # unwrap_model=unwrap_model,
        callback=train_callback,
        )
    return os.path.join(lora_path, lora_name+".safetensors")


# model_p = "/data/qll/stable-diffusion-webui/models"
# trigger_word = f"xzxzai"
# undesired_tags = ""  # 待测试五官
# train_data_dir = "/data/qll/pics/yijian_sanciyuan_train/test"
# train_auto(
#     train_data_dir=train_data_dir,  # 训练的图片路径
#     train_type=0,  # 训练的类别
#     task_id="zzlll",   # 任务id,作为Lora名称
#     sd_model_path=MODEL_PATH, # 底模路径
#     lora_path=LORA_PATH, # 文件夹名字
#     general_model_path=model_p, # 通用路径,
#     train_callback=train_callback, # callback函数
#     other_args=None # 预留一个，以备用
# )
