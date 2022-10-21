from dataclasses import dataclass, field
from typing import List

# set combo box to error msg instead of blank when cannot retrieve options from backend
ERROR_MSG = "Retrieval Failed"

# Used for status bar
STATE_READY = "Ready"
STATE_INIT = "Initializing..."
STATE_URLERROR = "Cannot reach backend"
STATE_RESET_DEFAULT = "All settings reset"
STATE_WAIT = "Please wait..."
STATE_TXT2IMG = "txt2img done!"
STATE_IMG2IMG = "img2img done!"
STATE_INPAINT = "inpaint done!"
STATE_UPSCALE = "upscale done!"

# Other currently hardcoded stuff
GET_CONFIG_TIMEOUT = 2  # 2 second timeout as getting config should be near instant
POST_TIMEOUT = None  # post might take forever depending on batch size/count
REFRESH_INTERVAL = 10000  # 10 seconds between auto-config refresh
CFG_FOLDER = "krita"  # which folder in ~/.config to store config
CFG_NAME = "krita_diff_plugin"  # name of config file
# selection mask can only be added after image is added, so timeout is needed
ADD_MASK_TIMEOUT = 100


@dataclass(frozen=True)
class Defaults:
    base_url: str = "http://127.0.0.1:8000"
    just_use_yaml: bool = False
    create_mask_layer: bool = True
    delete_temp_files: bool = True
    fix_aspect_ratio: bool = True
    only_full_img_tiling: bool = True
    filter_nsfw: bool = False
    do_exact_steps: bool = True

    # TODO: consider what to do with these
    new_img_path: str = "debug_image.png"
    new_img_mask_path: str = "debug_image_mask.png"

    sd_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    sd_model: str = "model.ckpt"
    sd_batch_size: int = 1
    sd_batch_count: int = 1
    sd_base_size: int = 512
    sd_max_size: int = 768
    sd_tiling: bool = False
    upscaler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    upscaler_name: str = "None"
    face_restorer_model_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    face_restorer_model: str = "None"
    codeformer_weight: float = 0.5
    include_grid: bool = False

    txt2img_prompt: str = ""
    txt2img_negative_prompt: str = ""
    txt2img_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    txt2img_sampler: str = "Euler a"
    txt2img_steps: int = 20
    txt2img_cfg_scale: float = 7.0
    txt2img_denoising_strength: float = 0.7
    txt2img_seed: str = ""
    txt2img_highres: bool = False
    # TODO: Seed variation

    img2img_prompt: str = ""
    img2img_negative_prompt: str = ""
    img2img_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
    img2img_sampler: str = "Euler a"
    img2img_steps: int = 40
    img2img_cfg_scale: float = 12.0
    img2img_denoising_strength: float = 0.8
    img2img_seed: str = ""
    img2img_color_correct: bool = False

    inpaint_prompt: str = ""
    inpaint_negative_prompt: str = ""
    inpaint_sampler_list: List[str] = field(default_factory=lambda: [ERROR_MSG])
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
    inpaint_full_res: bool = False
    inpaint_full_res_padding: int = 32
    inpaint_color_correct: bool = True

    upscale_upscaler_name: str = "None"
    upscale_downscale_first: bool = False


DEFAULTS = Defaults()
