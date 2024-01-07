import diffusers.pipelines as p

def is_sd15(model):
    if model is None:
        return False
    return isinstance(model, p.StableDiffusionPipeline) or isinstance(model, p.StableDiffusionImg2ImgPipeline) or isinstance(model, p.StableDiffusionInpaintPipeline)

def is_sdxl(model):
    if model is None:
        return False
    return isinstance(model, p.StableDiffusionXLPipeline) or isinstance(model, p.StableDiffusionXLImg2ImgPipeline) or isinstance(model, p.StableDiffusionXLInpaintPipeline)
