from PIL import Image
from modules.control.util import HWC3, resize_image
from modules import devices
from .marigold_pipeline import MarigoldPipeline


class MarigoldDetector:
    def __init__(self, model):
        self.model: MarigoldPipeline = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, cache_dir=None, **load_config):
        model = MarigoldPipeline.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, **load_config)
        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(
        self,
        input_image: Image,
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        color_map: str = "Spectral",
        output_type=None,
    ):
        self.model.to(device=devices.device, dtype=devices.dtype)
        res = self.model(
            input_image,
            denoising_steps=denoising_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=match_input_res,
            color_map=color_map if color_map != 'None' else 'Spectral',
            batch_size=1,
            show_progress_bar=True,
        )
        depth_map = res.depth_np
        depth_colored = res.depth_colored

        if output_type == "pil":
            depth_map = Image.fromarray(depth_map)

        return depth_colored if color_map != 'None' else depth_map
