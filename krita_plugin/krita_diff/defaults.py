from dataclasses import dataclass, field
from typing import List


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

    sd_model_list: List[str] = field(default_factory=list)
    sd_model: str = "model.ckpt"
    face_restorer_model_list: List[str] = field(default_factory=list)
    face_restorer_model: str = "CodeFormer"
    codeformer_weight: float = 0.5

    txt2img_prompt: str = ""
    txt2img_negative_prompt: str = ""
    txt2img_sampler_list: List[str] = field(default_factory=list)
    txt2img_sampler: str = "Euler a"
    txt2img_steps: int = 20
    txt2img_cfg_scale: float = 7.5
    txt2img_denoising_strength: float = 0.7
    txt2img_batch_count: int = 1
    txt2img_batch_size: int = 1
    txt2img_base_size: int = 512
    txt2img_max_size: int = 768
    txt2img_seed: str = ""
    txt2img_restore_faces: bool = False
    txt2img_tiling: bool = False

    img2img_prompt: str = ""
    img2img_negative_prompt: str = ""
    img2img_sampler_list: List[str] = field(default_factory=list)
    img2img_sampler: str = "Euler a"
    img2img_steps: int = 50
    img2img_cfg_scale: float = 12.0
    img2img_denoising_strength: float = 0.40
    img2img_batch_count: int = 1
    img2img_batch_size: int = 1
    img2img_base_size: int = 512
    img2img_max_size: int = 768
    img2img_seed: str = ""
    img2img_restore_faces: bool = False
    img2img_tiling: bool = False
    img2img_invert_mask: bool = False
    img2img_upscaler_name: str = "None"

    upscaler_list: List[str] = field(default_factory=list)
    upscale_upscaler_name: str = "None"
    upscale_downscale_first: bool = False
