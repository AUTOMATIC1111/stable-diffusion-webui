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
    return {
        'Autodetect': None,
        'Stable Diffusion': diffusers.StableDiffusionPipeline,
        'Stable Diffusion Img2Img': diffusers.StableDiffusionImg2ImgPipeline,
        'Stable Diffusion Instruct': diffusers.StableDiffusionInstructPix2PixPipeline,
        'Stable Diffusion Upscale': diffusers.StableDiffusionUpscalePipeline,
        'Stable Diffusion XL': diffusers.StableDiffusionXLPipeline,
        'Stable Diffusion XL Img2Img': diffusers.StableDiffusionXLImg2ImgPipeline,
        'Stable Diffusion XL Inpaint': diffusers.StableDiffusionXLInpaintPipeline,
        'Stable Diffusion XL Instruct': diffusers.StableDiffusionXLInstructPix2PixPipeline,
        # 'Kandinsky V1', 'Kandinsky V2', 'DeepFloyd IF', 'Shap-E', 'Kandinsky V1 Img2Img', 'Kandinsky V2 Img2Img', 'DeepFloyd IF Img2Img', 'Shap-E Img2Img',
    }
