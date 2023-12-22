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
