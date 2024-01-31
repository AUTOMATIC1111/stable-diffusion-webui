import os
import sys
import cv2
from base64 import b64encode
from pathlib import Path

import requests

BASE_URL = "http://localhost:7860"


# setup_test_env
os.environ['IGNORE_CMD_ARGS_ERRORS'] = 'True'

file_path = Path(__file__).resolve()
ext_root = file_path.parent.parent
a1111_root = ext_root.parent.parent

for p in (ext_root, a1111_root):
    if p not in sys.path:
        sys.path.append(str(p))

# Initialize A1111
from modules import initialize
initialize.imports()
initialize.initialize()

from scripts.enums import StableDiffusionVersion

def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img


def get_model(model_name: str, sd_version: StableDiffusionVersion = StableDiffusionVersion.SD1x) -> str:
    """ Find an available model with specified model name and sd_version. """
    if model_name.lower() == "none":
        return "None"

    r = requests.get(BASE_URL+"/controlnet/model_list")
    result = r.json()
    if "model_list" not in result:
        raise ValueError("No model available")

    candidates = [
        model
        for model in result["model_list"]
        if (
            model_name.lower() in model.lower() and 
            StableDiffusionVersion.detect_from_model_name(model) == sd_version
        )
    ]

    if not candidates:
        raise ValueError("No suitable model available")
    
    return candidates[0]
    

def get_modules():
    return requests.get(f"{BASE_URL}/controlnet/module_list").json()


def detect(json):
    return requests.post(BASE_URL+"/controlnet/detect", json=json)
