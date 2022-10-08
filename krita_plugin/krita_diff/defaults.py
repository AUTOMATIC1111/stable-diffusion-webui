from dataclasses import dataclass, field
from typing import List

# set combo box to error msg instead of blank when cannot retrieve options from backend
error_msg = "Retrieval Failed"


@dataclass(frozen=True)
class Defaults:
    base_url: str = "http://127.0.0.1:8000"
    just_use_yaml: bool = False
    create_mask_layer: bool = True
    delete_temp_files: bool = True
    workaround_timeout: int = 100
    png_quality: int = -1
    fix_aspect_ratio: bool = True
    only_full_img_tiling: bool = True
    filter_nsfw: bool = False
    color_correct: bool = True
    do_exact_steps: bool = True

    sd_model_list: List[str] = field(default_factory=lambda: [error_msg])
    sd_model: str = "model.ckpt"
    sd_batch_size: int = 1
    sd_batch_count: int = 1
    sd_base_size: int = 512
    sd_max_size: int = 768
    sd_tiling: bool = False
    upscaler_list: List[str] = field(default_factory=lambda: [error_msg])
    upscaler_name: str = "None"
    face_restorer_model_list: List[str] = field(default_factory=lambda: [error_msg])
    face_restorer_model: str = "CodeFormer"
    codeformer_weight: float = 0.5

    txt2img_prompt: str = ""
    txt2img_negative_prompt: str = ""
    txt2img_sampler_list: List[str] = field(default_factory=lambda: [error_msg])
    txt2img_sampler: str = "Euler a"
    txt2img_steps: int = 20
    txt2img_cfg_scale: float = 7.0
    txt2img_denoising_strength: float = 0.7
    txt2img_seed: str = ""
    txt2img_highres: bool = False
    # txt2img_scale_latent: bool = None
    # TODO: Seed variation

    img2img_prompt: str = ""
    img2img_negative_prompt: str = ""
    img2img_sampler_list: List[str] = field(default_factory=lambda: [error_msg])
    img2img_sampler: str = "Euler a"
    img2img_steps: int = 40
    img2img_cfg_scale: float = 12.0
    img2img_denoising_strength: float = 0.8
    img2img_seed: str = ""

    inpaint_prompt: str = ""
    inpaint_negative_prompt: str = ""
    inpaint_sampler_list: List[str] = field(default_factory=lambda: [error_msg])
    inpaint_sampler: str = "LMS"
    inpaint_steps: int = 100
    inpaint_cfg_scale: float = 5.0
    inpaint_denoising_strength: float = 0.40
    inpaint_seed: str = ""
    inpaint_invert_mask: bool = False
    inpaint_mask_blur: int = 4
    inpaint_fill_list: List[str] = field(
        # NOTE: list order corresponds to number to use in internal API!!!
        default_factory=lambda: ["blur", "preserve", "latent noise", "latent empty"]
    )
    inpaint_fill: str = "preserve"
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 32

    upscale_upscaler_name: str = "None"
    upscale_downscale_first: bool = False
