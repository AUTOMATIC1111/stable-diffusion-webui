import torch
import diffusers
import onnxruntime as ort
from typing import Any, Dict, Optional


initialized = False
submodels_sd = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)
submodels_sdxl = ("text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder",)
submodels_sdxl_refiner = ("text_encoder_2", "unet", "vae_encoder", "vae_decoder",)


class DynamicSessionOptions(ort.SessionOptions):
    config: Optional[Dict] = None

    def __init__(self):
        super().__init__()

        self.enable_mem_pattern = False

    @classmethod
    def from_sess_options(cls, sess_options: ort.SessionOptions):
        if isinstance(sess_options, DynamicSessionOptions):
            return sess_options.copy()
        return DynamicSessionOptions()

    def enable_static_dims(self, config: Dict):
        self.config = config
        self.add_free_dimension_override_by_name("unet_sample_batch", config["hidden_batch_size"])
        self.add_free_dimension_override_by_name("unet_sample_channels", 4)
        self.add_free_dimension_override_by_name("unet_sample_height", config["height"] // 8)
        self.add_free_dimension_override_by_name("unet_sample_width", config["width"] // 8)
        self.add_free_dimension_override_by_name("unet_time_batch", 1)
        self.add_free_dimension_override_by_name("unet_hidden_batch", config["hidden_batch_size"])
        self.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        if config["is_sdxl"] and not config["is_refiner"]:
            self.add_free_dimension_override_by_name("unet_text_embeds_batch", config["hidden_batch_size"])
            self.add_free_dimension_override_by_name("unet_text_embeds_size", 1280)
            self.add_free_dimension_override_by_name("unet_time_ids_batch", config["hidden_batch_size"])
            self.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    def copy(self):
        sess_options = DynamicSessionOptions()
        if self.config is not None:
            sess_options.enable_static_dims(self.config)
        return sess_options


class OnnxFakeModule:
    device = torch.device("cpu")
    dtype = torch.float32

    def to(self, *args, **kwargs):
        return self

    def type(self, *args, **kwargs):
        return self


class OnnxTemporalModel(OnnxFakeModule):
    """
    Replace the models which are not able to be moved to CPU.
    """
    previous_provider: Any
    path: str
    sess_options: ort.SessionOptions

    def __init__(self, previous_provider: Any, path: str, sess_options: ort.SessionOptions):
        self.previous_provider = previous_provider
        self.path = path
        self.sess_options = sess_options

    def to(self, *args, **kwargs):
        from modules.onnx_utils import extract_device

        device = extract_device(args, kwargs)
        if device is not None and device.type != "cpu":
            from modules.onnx_ep import TORCH_DEVICE_TO_EP

            provider = TORCH_DEVICE_TO_EP[device.type] if device.type in TORCH_DEVICE_TO_EP else self.previous_provider
            return OnnxRuntimeModel.load_model(self.path, provider, DynamicSessionOptions.from_sess_options(self.sess_options))
        return self


class OnnxRuntimeModel(OnnxFakeModule, diffusers.OnnxRuntimeModel):
    config = {} # dummy

    def named_modules(self): # dummy
        return ()

    def to(self, *args, **kwargs):
        from modules.onnx_utils import extract_device, move_inference_session

        device = extract_device(args, kwargs)
        if device is not None:
            self.device = device
            self.model = move_inference_session(self.model, device)
        return self


def preprocess_pipeline(p, refiner_enabled: bool):
    from modules import shared, sd_models

    if "ONNX" not in shared.opts.diffusers_pipeline:
        shared.log.warning(f"Unsupported pipeline for 'olive-ai' compile backend: {shared.opts.diffusers_pipeline}. You should select one of the ONNX pipelines.")
        return

    if shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "olive-ai":
        compile_height = p.height
        compile_width = p.width
        if (shared.compiled_model_state is None or
        shared.compiled_model_state.height != compile_height
        or shared.compiled_model_state.width != compile_width
        or shared.compiled_model_state.batch_size != p.batch_size):
            shared.log.info("Olive: Parameter change detected")
            shared.log.info("Olive: Recompiling base model")
            sd_models.unload_model_weights(op='model')
            sd_models.reload_model_weights(op='model')
            if refiner_enabled:
                shared.log.info("Olive: Recompiling refiner")
                sd_models.unload_model_weights(op='refiner')
                sd_models.reload_model_weights(op='refiner')
        shared.compiled_model_state.height = compile_height
        shared.compiled_model_state.width = compile_width
        shared.compiled_model_state.batch_size = p.batch_size

    if hasattr(shared.sd_model, "preprocess"):
        shared.sd_model = shared.sd_model.preprocess(p)
    if hasattr(shared.sd_refiner, "preprocess"):
        if shared.opts.onnx_unload_base:
            sd_models.unload_model_weights(op='model')
        shared.sd_refiner = shared.sd_refiner.preprocess(p)
        if shared.opts.onnx_unload_base:
            sd_models.reload_model_weights(op='model')
            shared.sd_model = shared.sd_model.preprocess(p)


def initialize():
    global initialized

    if initialized:
        return

    from modules import onnx_pipelines as pipelines

    # OnnxRuntimeModel Hijack.
    OnnxRuntimeModel.__module__ = 'diffusers'
    diffusers.OnnxRuntimeModel = OnnxRuntimeModel

    diffusers.OnnxStableDiffusionPipeline = pipelines.OnnxStableDiffusionPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionPipeline

    diffusers.OnnxStableDiffusionImg2ImgPipeline = pipelines.OnnxStableDiffusionImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline

    diffusers.OnnxStableDiffusionInpaintPipeline = pipelines.OnnxStableDiffusionInpaintPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionInpaintPipeline

    diffusers.OnnxStableDiffusionXLPipeline = pipelines.OnnxStableDiffusionXLPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline

    diffusers.OnnxStableDiffusionXLImg2ImgPipeline = pipelines.OnnxStableDiffusionXLImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline

    # Huggingface model compatibility
    diffusers.ORTStableDiffusionXLPipeline = diffusers.OnnxStableDiffusionXLPipeline
    diffusers.ORTStableDiffusionXLImg2ImgPipeline = diffusers.OnnxStableDiffusionXLImg2ImgPipeline

    initialized = True
