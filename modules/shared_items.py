def postprocessing_scripts():
    import modules.scripts
    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae
    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae
    modules.sd_vae.refresh_vae_list()


def list_crossattention():
    return [
        "Disabled",
        "xFormers",
        "Scaled-Dot-Product",
        "Doggettx's",
        "InvokeAI's",
        "Sub-quadratic",
        "Split attention"
    ]

def get_pipelines():
    import diffusers
    from installer import log
    pipelines = {
        'Autodetect': None,
        'Stable Diffusion': getattr(diffusers, 'StableDiffusionPipeline', None),
        'Stable Diffusion Img2Img': getattr(diffusers, 'StableDiffusionImg2ImgPipeline', None),
        'Stable Diffusion Instruct': getattr(diffusers, 'StableDiffusionInstructPix2PixPipeline', None),
        'Stable Diffusion Upscale': getattr(diffusers, 'StableDiffusionUpscalePipeline', None),
        'Stable Diffusion XL': getattr(diffusers, 'StableDiffusionXLPipeline', None),
        'Stable Diffusion XL Img2Img': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
        'Stable Diffusion XL Inpaint': getattr(diffusers, 'StableDiffusionXLInpaintPipeline', None),
        'Stable Diffusion XL Instruct': getattr(diffusers, 'StableDiffusionXLInstructPix2PixPipeline', None),
        'Custom Diffusers Pipeline': getattr(diffusers, 'DiffusionPipeline', None),
        # 'Test': getattr(diffusers, 'TestPipeline', None),
        # 'Kandinsky V1', 'Kandinsky V2', 'DeepFloyd IF', 'Shap-E', 'Kandinsky V1 Img2Img', 'Kandinsky V2 Img2Img', 'DeepFloyd IF Img2Img', 'Shap-E Img2Img',
    }
    for k, v in pipelines.items():
        if k != 'Autodetect' and v is None:
            log.error(f'Not available: pipeline={k} diffusers={diffusers.__version__} path={diffusers.__file__}')
    return pipelines
