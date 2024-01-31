import os
import re
from typing import List

import numpy as np
import torch
from torch.cuda import nvtx
from polygraphy.logger import G_LOGGER
import gradio as gr

from modules import script_callbacks, sd_unet, devices, scripts, shared

import ui_trt
from utilities import Engine
from model_manager import TRT_MODEL_DIR, modelmanager
from datastructures import ModelType
from scripts.lora import apply_loras

G_LOGGER.module_severity = G_LOGGER.ERROR


class TrtUnetOption(sd_unet.SdUnetOption):
    def __init__(self, name: str, filename: List[dict]):
        self.label = f"[TRT] {name}"
        self.model_name = name
        self.configs = filename

    def create_unet(self):
        return TrtUnet(self.model_name, self.configs)


class TrtUnet(sd_unet.SdUnet):
    def __init__(self, model_name: str, configs: List[dict], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stream = None
        self.model_name = model_name
        self.configs = configs

        self.profile_idx = 0
        self.loaded_config = None

        self.engine_vram_req = 0
        self.refitted_keys = set()

        self.engine = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        nvtx.range_push("forward")
        feed_dict = {
            "sample": x.float(),
            "timesteps": timesteps.float(),
            "encoder_hidden_states": context.float(),
        }
        if "y" in kwargs:
            feed_dict["y"] = kwargs["y"].float()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=devices.device
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["latent"]

        nvtx.range_pop()
        return out

    def apply_loras(self, refit_dict: dict):
        if not self.refitted_keys.issubset(set(refit_dict.keys())):
            # Need to ensure that weights that have been modified before and are not present anymore are reset.
            self.refitted_keys = set()
            self.switch_engine()

        self.engine.refit_from_dict(refit_dict, is_fp16=True)
        self.refitted_keys = set(refit_dict.keys())

    def switch_engine(self):
        self.loaded_config = self.configs[self.profile_idx]
        self.engine.reset(os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"]))
        self.activate()

    def activate(self):
        self.loaded_config = self.configs[self.profile_idx]
        if self.engine is None:
            self.engine = Engine(
                os.path.join(TRT_MODEL_DIR, self.loaded_config["filepath"])
            )
        self.engine.load()
        print(f"\nLoaded Profile: {self.profile_idx}")
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

    def deactivate(self):
        del self.engine


class TensorRTScript(scripts.Script):
    def __init__(self) -> None:
        self.loaded_model = None
        self.lora_hash = ""
        self.update_lora = False
        self.lora_refit_dict = {}
        self.idx = None
        self.hr_idx = None
        self.torch_unet = False

    def title(self):
        return "TensorRT"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def setup(self, p, *args):
        return super().setup(p, *args)

    def before_process(self, p, *args):  # 1
        # Check divisibilty
        if p.width % 64 or p.height % 64:
            gr.Error("Target resolution must be divisible by 64 in both dimensions.")

        if self.is_img2img:
            return
        if p.enable_hr:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            if hr_w % 64 or hr_h % 64:
                gr.Error(
                    "HIRES Fix resolution must be divisible by 64 in both dimensions. Please change the upscale factor or disable HIRES Fix."
                )

    def get_profile_idx(self, p, model_name: str, model_type: ModelType) -> (int, int):
        best_hr = None

        if self.is_img2img:
            hr_scale = 1
        else:
            hr_scale = p.hr_scale if p.enable_hr else 1
        (
            valid_models,
            distances,
            idx,
        ) = modelmanager.get_valid_models(
            model_name,
            p.width,
            p.height,
            p.batch_size,
            77,  # model_type
        )  # TODO: max_embedding, just ignore?
        if len(valid_models) == 0:
            gr.Error(
                f"""No valid profile found for ({model_name}) LOWRES. Please go to the TensorRT tab and generate an engine with the necessary profile. 
                If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
            )
            return None, None
        best = idx[np.argmin(distances)]
        best_hr = best

        if hr_scale != 1:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
            valid_models_hr, distances_hr, idx_hr = modelmanager.get_valid_models(
                model_name,
                hr_w,
                hr_h,
                p.batch_size,
                77,  # model_type
            )  # TODO: max_embedding
            if len(valid_models_hr) == 0:
                gr.Error(
                    f"""No valid profile found for ({model_name}) HIRES. Please go to the TensorRT tab and generate an engine with the necessary profile. 
                    If using hires.fix, you need an engine for both the base and upscaled resolutions. Otherwise, use the default (torch) U-Net."""
                )
            merged_idx = [i for i, id in enumerate(idx) if id in idx_hr]
            if len(merged_idx) == 0:
                gr.Warning(
                    "No model available for both ({}) LOWRES ({}x{}) and HIRES ({}x{}). This will slow-down inference.".format(
                        model_name, p.width, p.height, hr_w, hr_h
                    )
                )
                return None, None
            else:
                _distances = [distances[i] for i in merged_idx]
                best_hr = merged_idx[np.argmin(_distances)]
                best = best_hr

        return best, best_hr

    def get_loras(self, p):
        lora_pathes = []
        lora_scales = []

        # get lora from prompt
        _prompt = p.prompt
        extra_networks = re.findall("\<(.*?)\>", _prompt)
        loras = [net for net in extra_networks if net.startswith("lora")]

        # Avoid that extra networks will be loaded
        for lora in loras:
            _prompt = _prompt.replace(f"<{lora}>", "")
        p.prompt = _prompt

        # check if lora config has changes
        if self.lora_hash != "".join(loras):
            self.lora_hash = "".join(loras)
            self.update_lora = True
            if self.lora_hash == "":
                self.lora_refit_dict = {}
                return
        else:
            return

        # Get pathes
        print("Apllying LoRAs: " + str(loras))
        available = modelmanager.available_loras()
        for lora in loras:
            lora_name, lora_scale = lora.split(":")[1:]
            lora_scales.append(float(lora_scale))
            if lora_name not in available:
                raise Exception(
                    f"Please export the LoRA checkpoint {lora_name} first from the TensorRT LoRA tab"
                )
            lora_pathes.append(
                available[lora_name]
            )

        # Merge lora refit dicts
        base_name, base_path = modelmanager.get_onnx_path(p.sd_model_name)
        refit_dict = apply_loras(base_path, lora_pathes, lora_scales)

        self.lora_refit_dict = refit_dict

    def process(self, p, *args):
        # before unet_init
        sd_unet_option = sd_unet.get_unet_option()
        if sd_unet_option is None:
            return

        if not sd_unet_option.model_name == p.sd_model_name:
            gr.Error(
                """Selected torch model ({}) does not match the selected TensorRT U-Net ({}). 
                Please ensure that both models are the same or select Automatic from the SD UNet dropdown.""".format(
                    p.sd_model_name, sd_unet_option.model_name
                )
            )
        self.idx, self.hr_idx = self.get_profile_idx(p, p.sd_model_name, ModelType.UNET)
        self.torch_unet = self.idx is None or self.hr_idx is None

        try:
            if not self.torch_unet:
                self.get_loras(p)
        except Exception as e:
            gr.Error(e)
            raise e

        self.apply_unet(sd_unet_option)

    def apply_unet(self, sd_unet_option):
        if (
            sd_unet_option == sd_unet.current_unet_option
            and sd_unet.current_unet is not None
            and not self.torch_unet
        ):
            return

        if sd_unet.current_unet is not None:
            sd_unet.current_unet.deactivate()

        if self.torch_unet:
            gr.Warning("Enabling PyTorch fallback as no engine was found.")
            sd_unet.current_unet = None
            sd_unet.current_unet_option = sd_unet_option
            shared.sd_model.model.diffusion_model.to(devices.device)
            return
        else:
            shared.sd_model.model.diffusion_model.to(devices.cpu)
            devices.torch_gc()
            if self.lora_refit_dict:
                self.update_lora = True
        sd_unet.current_unet = sd_unet_option.create_unet()
        sd_unet.current_unet.profile_idx = self.idx
        sd_unet.current_unet.option = sd_unet_option
        sd_unet.current_unet_option = sd_unet_option

        print(f"Activating unet: {sd_unet.current_unet.option.label}")
        sd_unet.current_unet.activate()

    def process_batch(self, p, *args, **kwargs):
        # Called for each batch count
        if self.torch_unet:
            return super().process_batch(p, *args, **kwargs)

        if self.idx != sd_unet.current_unet.profile_idx:
            sd_unet.current_unet.profile_idx = self.idx
            sd_unet.current_unet.switch_engine()

    def before_hr(self, p, *args):
        if self.idx != self.hr_idx:
            sd_unet.current_unet.profile_idx = self.hr_idx
            sd_unet.current_unet.switch_engine()

        return super().before_hr(p, *args)  # 4 (Only when HR starts.....)

    def after_extra_networks_activate(self, p, *args, **kwargs):
        if self.update_lora and not self.torch_unet:
            self.update_lora = False
            sd_unet.current_unet.apply_loras(self.lora_refit_dict)


def list_unets(l):
    model = modelmanager.available_models()
    for k, v in model.items():
        if v[0]["config"].lora:
            continue
        label = "{} ({})".format(k, v[0]["base_model"]) if v[0]["config"].lora else k
        l.append(TrtUnetOption(label, v))


script_callbacks.on_list_unets(list_unets)
script_callbacks.on_ui_tabs(ui_trt.on_ui_tabs)
