from typing import Optional, Dict, Any
import onnxruntime as ort
import optimum.onnxruntime
from modules.onnx_impl.pipelines import CallablePipelineBase
from modules.onnx_impl.pipelines.utils import prepare_latents


class OnnxStableDiffusionXLPipeline(CallablePipelineBase, optimum.onnxruntime.ORTStableDiffusionXLPipeline):
    __module__ = 'optimum.onnxruntime.modeling_diffusion'
    __name__ = 'ORTStableDiffusionXLPipeline'

    def __init__(
        self,
        vae_decoder: ort.InferenceSession,
        text_encoder: ort.InferenceSession,
        unet: ort.InferenceSession,
        config: Dict[str, Any],
        tokenizer: Any,
        scheduler: Any,
        feature_extractor: Any = None,
        vae_encoder: Optional[ort.InferenceSession] = None,
        text_encoder_2: Optional[ort.InferenceSession] = None,
        tokenizer_2: Any = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir = None,
        add_watermarker: Optional[bool] = None
    ):
        optimum.onnxruntime.ORTStableDiffusionXLPipeline.__init__(self, vae_decoder, text_encoder, unet, config, tokenizer, scheduler, feature_extractor, vae_encoder, text_encoder_2, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)
        super().__init__()
        del self.image_processor # This image processor requires np array. In order to share same workflow with non-XL pipelines, delete it.

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        return prepare_latents(self.scheduler.init_noise_sigma, batch_size, height, width, dtype, generator, latents, num_channels_latents, self.vae_scale_factor)
