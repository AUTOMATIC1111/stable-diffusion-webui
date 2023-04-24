

def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


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
        "Disable cross-attention layer optimization",
        "xFormers",
        "Scaled-Dot-Product",
        "Doggettx's",
        "InvokeAI's",
        "Sub-quadratic",
        "Split attention"
    ]
# parser.add_argument("--sub-quad-q-chunk-size", type=int, help="query chunk size for the sub-quadratic cross-attention layer optimization to use", default=1024)
# parser.add_argument("--sub-quad-kv-chunk-size", type=int, help="kv chunk size for the sub-quadratic cross-attention layer optimization to use", default=None)
# parser.add_argument("--sub-quad-chunk-threshold", type=int, help="the percentage of VRAM threshold for the sub-quadratic cross-attention layer optimization to use chunking", default=None)
