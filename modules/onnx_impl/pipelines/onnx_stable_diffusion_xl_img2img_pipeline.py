from typing import Optional, Dict, Any
import numpy as np
import torch
import onnxruntime as ort
import optimum.onnxruntime
from modules.onnx_impl.pipelines import CallablePipelineBase
from modules.onnx_impl.pipelines.utils import randn_tensor


class OnnxStableDiffusionXLImg2ImgPipeline(CallablePipelineBase, optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline):
    __module__ = 'optimum.onnxruntime.modeling_diffusion'
    __name__ = 'ORTStableDiffusionXLImg2ImgPipeline'

    def __init__(
        self,
        vae_decoder: ort.InferenceSession,
        text_encoder: ort.InferenceSession,
        unet: ort.InferenceSession,
        config: Dict[str, Any],
        tokenizer: Any,
        scheduler: Any,
        feature_extractor = None,
        vae_encoder: Optional[ort.InferenceSession] = None,
        text_encoder_2: Optional[ort.InferenceSession] = None,
        tokenizer_2: Any = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir = None,
        add_watermarker: Optional[bool] = None
    ):
        optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline.__init__(self, vae_decoder, text_encoder, unet, config, tokenizer, scheduler, feature_extractor, vae_encoder, text_encoder_2, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)
        super().__init__()
        del self.image_processor # This image processor requires np array. In order to share same workflow with non-XL pipelines, delete it.

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            init_latents = self.vae_encoder(sample=image)[0] * self.vae_decoder.config.get("scaling_factor", 0.18215)

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = np.concatenate([init_latents] * additional_image_per_prompt, axis=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = np.concatenate([init_latents], axis=0)

        # add noise to latents using the timesteps
        noise = randn_tensor(init_latents.shape, dtype, generator)
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timestep)
        )
        return init_latents.numpy()
