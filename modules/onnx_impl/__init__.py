import os
from typing import Any, Dict, Optional
import numpy as np
import torch
import diffusers
import onnxruntime as ort
import optimum.onnxruntime


initialized = False
run_olive_workflow = None


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


class TorchCompatibleModule:
    device = torch.device("cpu")
    dtype = torch.float32

    def to(self, *_, **__):
        raise NotImplementedError

    def type(self, *_, **__):
        return self


class TemporalModule(TorchCompatibleModule):
    """
    Replace the models which are not able to be moved to CPU.
    """
    provider: Any
    path: str
    sess_options: ort.SessionOptions

    def __init__(self, provider: Any, path: str, sess_options: ort.SessionOptions):
        self.provider = provider
        self.path = path
        self.sess_options = sess_options

    def to(self, *args, **kwargs):
        from .utils import extract_device

        device = extract_device(args, kwargs)
        if device is not None and device.type != "cpu":
            from .execution_providers import TORCH_DEVICE_TO_EP

            provider = TORCH_DEVICE_TO_EP[device.type] if device.type in TORCH_DEVICE_TO_EP else self.provider
            return OnnxRuntimeModel.load_model(self.path, provider, DynamicSessionOptions.from_sess_options(self.sess_options))
        return self


class OnnxRuntimeModel(TorchCompatibleModule, diffusers.OnnxRuntimeModel):
    config = {} # dummy

    def named_modules(self): # dummy
        return ()

    def to(self, *args, **kwargs):
        from modules.onnx_impl.utils import extract_device, move_inference_session

        device = extract_device(args, kwargs)
        if device is not None:
            self.device = device
            self.model = move_inference_session(self.model, device)
        return self


class VAEConfig:
    DEFAULTS = {
        "scaling_factor": 0.18215,
    }

    config: Dict

    def __init__(self, config: Dict):
        self.config = config

    def __getattr__(self, key):
        return self.config.get(key, VAEConfig.DEFAULTS[key])


class VAE(TorchCompatibleModule):
    pipeline: Any

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    @property
    def config(self):
        return VAEConfig(self.pipeline.vae_decoder.config)

    @property
    def device(self):
        return self.pipeline.vae_decoder.device

    def encode(self, latent_sample: torch.Tensor, return_dict: bool): # pylint: disable=unused-argument
        latents_np = latent_sample.cpu().numpy()
        return [
            torch.from_numpy(np.concatenate(
                [self.pipeline.vae_encoder(latent_sample=latents_np[i : i + 1])[0] for i in range(latents_np.shape[0])]
            )).to(latent_sample.device)
        ]

    def decode(self, latent_sample: torch.Tensor, return_dict: bool): # pylint: disable=unused-argument
        latents_np = latent_sample.cpu().numpy()
        return [
            torch.from_numpy(np.concatenate(
                [self.pipeline.vae_decoder(latent_sample=latents_np[i : i + 1])[0] for i in range(latents_np.shape[0])]
            )).to(latent_sample.device)
        ]

    def to(self, *args, **kwargs):
        self.pipeline.vae_encoder = self.pipeline.vae_encoder.to(*args, **kwargs)
        self.pipeline.vae_decoder = self.pipeline.vae_decoder.to(*args, **kwargs)
        return self


def preprocess_pipeline(p, refiner_enabled: bool):
    from modules import shared, sd_models

    if "ONNX" not in shared.opts.diffusers_pipeline:
        shared.log.warning(f"Unsupported pipeline for 'olive-ai' compile backend: {shared.opts.diffusers_pipeline}. You should select one of the ONNX pipelines.")
        return shared.sd_model

    if shared.opts.cuda_compile_backend == "olive-ai" and len(shared.opts.cuda_compile) != 1:
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

    return shared.sd_model


def ORTDiffusionModelPart_to(self, *args, **kwargs):
    self.parent_model = self.parent_model.to(*args, **kwargs)
    return self


def initialize():
    global initialized # pylint: disable=global-statement

    if initialized:
        return

    from modules import devices
    from modules.paths import models_path
    from .execution_providers import ExecutionProvider, TORCH_DEVICE_TO_EP

    onnx_dir = os.path.join(models_path, "ONNX")
    if not os.path.isdir(onnx_dir):
        os.mkdir(onnx_dir)

    if devices.backend == "rocm":
        TORCH_DEVICE_TO_EP["cuda"] = ExecutionProvider.ROCm

    from .pipelines.onnx_stable_diffusion_pipeline import OnnxStableDiffusionPipeline
    from .pipelines.onnx_stable_diffusion_img2img_pipeline import OnnxStableDiffusionImg2ImgPipeline
    from .pipelines.onnx_stable_diffusion_inpaint_pipeline import OnnxStableDiffusionInpaintPipeline
    from .pipelines.onnx_stable_diffusion_upscale_pipeline import OnnxStableDiffusionUpscalePipeline
    from .pipelines.onnx_stable_diffusion_xl_pipeline import OnnxStableDiffusionXLPipeline
    from .pipelines.onnx_stable_diffusion_xl_img2img_pipeline import OnnxStableDiffusionXLImg2ImgPipeline

    # OnnxRuntimeModel Hijack.
    OnnxRuntimeModel.__module__ = 'diffusers'
    diffusers.OnnxRuntimeModel = OnnxRuntimeModel

    diffusers.OnnxStableDiffusionPipeline = OnnxStableDiffusionPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionPipeline

    diffusers.OnnxStableDiffusionImg2ImgPipeline = OnnxStableDiffusionImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline

    diffusers.OnnxStableDiffusionInpaintPipeline = OnnxStableDiffusionInpaintPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionInpaintPipeline

    diffusers.OnnxStableDiffusionUpscalePipeline = OnnxStableDiffusionUpscalePipeline

    diffusers.OnnxStableDiffusionXLPipeline = OnnxStableDiffusionXLPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline

    diffusers.OnnxStableDiffusionXLImg2ImgPipeline = OnnxStableDiffusionXLImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline

    # Huggingface model compatibility
    diffusers.ORTStableDiffusionXLPipeline = diffusers.OnnxStableDiffusionXLPipeline
    diffusers.ORTStableDiffusionXLImg2ImgPipeline = diffusers.OnnxStableDiffusionXLImg2ImgPipeline

    optimum.onnxruntime.modeling_diffusion._ORTDiffusionModelPart.to = ORTDiffusionModelPart_to # pylint: disable=protected-access

    initialized = True


def initialize_olive():
    global run_olive_workflow # pylint: disable=global-statement
    from installer import installed, log
    if not installed("olive-ai"):
        return
    import sys
    import importlib
    orig_sys_path = sys.path
    venv_dir = os.environ.get("VENV_DIR", os.path.join(os.getcwd(), 'venv'))
    try:
        spec = importlib.util.find_spec('onnxruntime.transformers')
        sys.path = [d for d in spec.submodule_search_locations + sys.path if sys.path[1] not in d or venv_dir in d]
        from onnxruntime.transformers import convert_generation # pylint: disable=unused-import
        spec = importlib.util.find_spec('olive')
        sys.path = spec.submodule_search_locations + sys.path
        run_olive_workflow = importlib.import_module('olive.workflows').run
    except Exception as e:
        run_olive_workflow = None
        log.error(f'Olive: Failed to load olive-ai: {e}')
    sys.path = orig_sys_path


def install_olive():
    from installer import installed, install, log

    if installed("olive-ai"):
        return

    try:
        log.info('Installing Olive')
        install('olive-ai', 'olive-ai', ignore=True)
        import olive.workflows # pylint: disable=unused-import
    except Exception as e:
        log.error(f'Olive: Failed to load olive-ai: {e}')
    else:
        log.info('Olive: Please restart webui session.')
