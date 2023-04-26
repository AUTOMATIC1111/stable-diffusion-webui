import os.path
from os import mkdir
from modules import paths, shared
from modules.sd_models import model_path as sd_model_dir_path
from basicsr.utils.download_util import load_file_from_url


def CheckModelsExist():
    print("witchpot initialization")

    # StableDiffusion
    sd_model_url = ""
    #sd_model_dir_path = os.path.abspath(os.path.join(paths.models_path, "Stable-diffusion"))
    sd_model_name = ""
    sd_model_path = os.path.abspath(os.path.join(sd_model_dir_path, sd_model_name))
    
    print("StableDiffusion_dir : " + sd_model_dir_path)        

    #if not os.path.exists(sd_model_path):
    #    if not os.path.exists(sd_model_dir_path):
    #        os.makedirs(sd_model_dir_path)

    #    load_file_from_url(sd_model_url, sd_model_dir_path, True, sd_model_name)

    # LoRA
    lora_model_url = "https://huggingface.co/Witchpot/icestage/resolve/main/witchpot-icestage-sd-1-5.safetensors"
    lora_models_dir_path = os.path.abspath(shared.cmd_opts.lora_dir)
    lora_model_name = ""
    lora_model_path = os.path.abspath(os.path.join(lora_models_dir_path, lora_model_name))

    print("LoRA_dir : " + lora_models_dir_path)

    #if not os.path.exists(lora_model_path):
    #    if not os.path.exists(lora_models_dir_path):
    #        os.makedirs(lora_models_dir_path)

    #    load_file_from_url(lora_model_url, lora_models_dir_path, True, lora_model_name)

    # ControlNet
    cn_model_url = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors"
    cn_models_dir_path = os.path.abspath(os.path.join(paths.models_path, "ControlNet"))
    cn_model_name = "control_v11f1p_sd15_depth_fp16.safetensors"
    cn_model_path = os.path.abspath(os.path.join(cn_models_dir_path, cn_model_name))

    print("ControlNet_dir : " + cn_models_dir_path)

    if not os.path.exists(cn_model_path):
        if not os.path.exists(cn_models_dir_path):
            os.makedirs(cn_models_dir_path)

        load_file_from_url(cn_model_url, cn_models_dir_path, True, cn_model_name)