

def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]

def postprocessing_scripts():
    import modules.scripts

    return modules.scripts.scripts_postproc.scripts