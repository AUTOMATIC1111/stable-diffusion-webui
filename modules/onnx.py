import torch
import diffusers


initialized = False
submodels_sd = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)
submodels_sdxl = ("text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder",)
submodels_sdxl_refiner = ("text_encoder_2", "unet", "vae_encoder", "vae_decoder",)


class OnnxFakeModule:
    device = torch.device("cpu")
    dtype = torch.float32

    def to(self, *args, **kwargs):
        return self

    def type(self, *args, **kwargs):
        return self


class OnnxRuntimeModel(OnnxFakeModule, diffusers.OnnxRuntimeModel):
    config = {} # dummy

    def named_modules(self): # dummy
        return ()


def optimize_pipeline(p, refiner_enabled: bool):
    from modules import shared, sd_models

    if "ONNX" not in shared.opts.diffusers_pipeline:
        shared.log.warning(f"Unsupported pipeline for 'olive-ai' compile backend: {shared.opts.diffusers_pipeline}. You should select one of the ONNX pipelines.")
        return

    if shared.opts.cuda_compile_backend == "olive-ai":
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


def initialize():
    global initialized

    if initialized:
        return

    from modules.onnx_pipelines import do_diffusers_hijack

    # OnnxRuntimeModel Hijack.
    OnnxRuntimeModel.__module__ = 'diffusers'
    diffusers.OnnxRuntimeModel = OnnxRuntimeModel

    do_diffusers_hijack()

    initialized = True
