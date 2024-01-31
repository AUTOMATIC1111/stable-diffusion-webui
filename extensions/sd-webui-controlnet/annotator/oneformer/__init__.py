import os
from modules import devices
from annotator.annotator_path import models_path
from .api import make_detectron2_model, semantic_run


class OneformerDetector:
    model_dir = os.path.join(models_path, "oneformer")
    configs = {
        "coco": {
            "name": "150_16_swin_l_oneformer_coco_100ep.pth",
            "config": 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml'
        },
        "ade20k": {
            "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
            "config": 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml'
        }
    }

    def __init__(self, config):
        self.model = None
        self.metadata = None
        self.config = config
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/" + self.config["name"]
        modelpath = os.path.join(self.model_dir, self.config["name"])
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        config = os.path.join(os.path.dirname(__file__), self.config["config"])
        model, self.metadata = make_detectron2_model(config, modelpath)
        self.model = model

    def unload_model(self):
        if self.model is not None:
            self.model.model.cpu()

    def __call__(self, img):
        if self.model is None:
            self.load_model()
            
        self.model.model.to(self.device)
        return semantic_run(img, self.model, self.metadata)
