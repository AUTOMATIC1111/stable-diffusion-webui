from typing_extensions import Literal, TypeAlias
import warnings
from modules import shared, scripts
import os

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from controlnet_aux.processor import (
        CannyDetector, MidasDetector, HEDdetector, LineartDetector, LineartAnimeDetector, OpenposeDetector
    )


Processor_id: TypeAlias = Literal[
    "canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "tile"
]

class Annotator:
    def __init__(self, processor_id: Processor_id, model_path=os.path.join(scripts.basedir(), "models")+"/videorendition/Annotators", detect_resolution=None):
        if processor_id == "canny":
            self.processor = CannyDetector()
        elif processor_id == "depth":
            self.processor = MidasDetector.from_pretrained(model_path)
        elif processor_id == "softedge":
            self.processor = HEDdetector.from_pretrained(model_path)
        elif processor_id == "lineart":
            self.processor = LineartDetector.from_pretrained(model_path)
        elif processor_id == "lineart_anime":
            self.processor = LineartAnimeDetector.from_pretrained(model_path)
        elif processor_id == "openpose":
            self.processor = OpenposeDetector.from_pretrained(model_path)
        elif processor_id == "tile":
            self.processor = None
        else:
            raise ValueError(f"Unsupported processor_id: {processor_id}")
        
        self.processor_id = processor_id
        self.detect_resolution = detect_resolution

    def __call__(self, image):
        width, height = image.size
        if self.processor_id == "openpose":
            kwargs = {
                "include_body": True,
                "include_hand": True,
                "include_face": True
            }
        else:
            kwargs = {}
        if self.processor is not None:
            detect_resolution = self.detect_resolution if self.detect_resolution is not None else min(width, height)
            image = self.processor(image, detect_resolution=detect_resolution, image_resolution=min(width, height), **kwargs)
        image = image.resize((width, height))
        return image

