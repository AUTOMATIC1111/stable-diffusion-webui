from ldm.models.diffusion.ddpm import LatentDiffusion
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from modules.sd_models import CheckpointInfo


class WebuiSdModel(LatentDiffusion):
    """This class is not actually instantinated, but its fields are created and fieeld by webui"""

    lowvram: bool
    """True if lowvram/medvram optimizations are enabled -- see modules.lowvram for more info"""

    sd_model_hash: str
    """short hash, 10 first characters of SHA1 hash of the model file; may be None if --no-hashing flag is used"""

    sd_model_checkpoint: str
    """path to the file on disk that model weights were obtained from"""

    sd_checkpoint_info: 'CheckpointInfo'
    """structure with additional information about the file with model's weights"""

    is_sdxl: bool
    """True if the model's architecture is SDXL or SSD"""

    is_ssd: bool
    """True if the model is SSD"""

    is_sd2: bool
    """True if the model's architecture is SD 2.x"""

    is_sd1: bool
    """True if the model's architecture is SD 1.x"""
