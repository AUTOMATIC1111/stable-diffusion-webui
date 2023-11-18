import os
from dataclasses import dataclass

import safetensors
import torch
from tensordict import TensorDict



@dataclass
class SDModel:
    model_path: os.PathLike
    device: torch.device

    def load_model(self):
        # logging.info(f"Loading: {self.model_path}")
        if os.path.splitext(self.model_path)[1] == ".safetensors":
            ckpt = safetensors.torch.load_file(
                self.model_path,
                device=self.device,
            )
        else:
            ckpt = torch.load(self.model_path, map_location=self.device)

        return TensorDict.from_dict(get_state_dict_from_checkpoint(ckpt))


# TODO: tidy up
# from: stable-diffusion-webui/modules/sd_models.py
def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        if new_key := transform_checkpoint_dict_key(k):
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


chckpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k
