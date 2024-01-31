import os
import cv2
import torch

from modules import devices
from modules.modelloader import load_file_from_url
from annotator.annotator_path import models_path
from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig, CLIPImageProcessor


config_clip_g = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 1664,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 48,
  "patch_size": 14,
  "projection_dim": 1280,
  "torch_dtype": "float32"
}

config_clip_h = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 1280,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 5120,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 32,
  "patch_size": 14,
  "projection_dim": 1024,
  "torch_dtype": "float32"
}

config_clip_vitl = {
  "attention_dropout": 0.0,
  "dropout": 0.0,
  "hidden_act": "quick_gelu",
  "hidden_size": 1024,
  "image_size": 224,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-05,
  "model_type": "clip_vision_model",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 24,
  "patch_size": 14,
  "projection_dim": 768,
  "torch_dtype": "float32"
}

configs = {
    'clip_g': config_clip_g,
    'clip_h': config_clip_h,
    'clip_vitl': config_clip_vitl,
}

downloads = {
    'clip_vitl': 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin',
    'clip_g': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/clip_g.pth',
    'clip_h': 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin'
}


clip_vision_h_uc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip_vision_h_uc.data')
clip_vision_h_uc = torch.load(clip_vision_h_uc,  map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['uc']

clip_vision_vith_uc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clip_vision_vith_uc.data')
clip_vision_vith_uc = torch.load(clip_vision_vith_uc, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['uc']


class ClipVisionDetector:
    def __init__(self, config, low_vram: bool):
        assert config in downloads
        self.download_link = downloads[config]
        self.model_path = os.path.join(models_path, 'clip_vision')
        self.file_name = config + '.pth'
        self.config = configs[config]
        self.device = (
            torch.device("cpu") if low_vram else
            devices.get_device_for("controlnet")
        )
        os.makedirs(self.model_path, exist_ok=True)
        file_path = os.path.join(self.model_path, self.file_name)
        if not os.path.exists(file_path):
            load_file_from_url(url=self.download_link, model_dir=self.model_path, file_name=self.file_name)
        config = CLIPVisionConfig(**self.config)

        self.model = CLIPVisionModelWithProjection(config)
        self.processor = CLIPImageProcessor(crop_size=224,
                                            do_center_crop=True,
                                            do_convert_rgb=True,
                                            do_normalize=True,
                                            do_resize=True,
                                            image_mean=[0.48145466, 0.4578275, 0.40821073],
                                            image_std=[0.26862954, 0.26130258, 0.27577711],
                                            resample=3,
                                            size=224)
        sd = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=False)
        del sd
        self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model.to('meta')

    def __call__(self, input_image):
        with torch.no_grad():
            input_image = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_AREA)
            feat = self.processor(images=input_image, return_tensors="pt")
            feat['pixel_values'] = feat['pixel_values'].to(self.device)
            result = self.model(**feat, output_hidden_states=True)
            result['hidden_states'] = [v.to(self.device) for v in result['hidden_states']]
            result = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        return result
