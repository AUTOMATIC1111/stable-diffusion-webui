import os
from modules import sd_samplers, shared, scripts, script_callbacks
from modules.script_callbacks import ImageSaveParams
import modules.images as images
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules.shared import opts, OptionInfo
from modules.paths import script_path

from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform
from launch import is_installed, run_pip

if platform.system() == "Windows" and not is_installed("pywin32"):
    run_pip(f"install pywin32", "pywin32")
try:
    from tools.add_tags import tag_files
except:
    print("Aesthetic Image Scorer: Unable to load Windows tagging script")
    tag_files = None

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


try:
    force_cpu = opts.ais_force_cpu
except:
    force_cpu = False

if force_cpu:
    print("Aesthtic Image Scorer: Forcing prediction model to run on CPU")
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"
# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device(device=device))

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features


def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()


def on_ui_settings():
    options = {}
    options.update(shared.options_section(('ais', "Aesthetic Image Scorer"), {
        "ais_add_exif": OptionInfo(False, "Save score as EXIF or PNG Info Chunk"),
        "ais_windows_tag": OptionInfo(False, "Save score as tag (Windows Only)"),
        "ais_force_cpu": OptionInfo(False, "Force CPU (Requires Custom Script Reload)"),
    }))

    opts.add_option("ais_add_exif", options["ais_add_exif"])
    opts.add_option("ais_windows_tag", options["ais_windows_tag"])
    opts.add_option("ais_force_cpu", options["ais_force_cpu"])


def on_before_image_saved(params: ImageSaveParams):
    if opts.ais_add_exif:
        score = round(get_score(params.image), 1)
        params.pnginfo.update({
            "aesthetic_score": score,
        })


def on_image_saved(params: ImageSaveParams):
    filename = os.path.realpath(os.path.join(script_path, params.filename))
    if "aesthetic_score" in params.pnginfo:
        score = params.pnginfo["aesthetic_score"]
    else:
        score = round(get_score(params.image), 1)
    if score is not None and opts.ais_windows_tag:
        if tag_files is not None:
            tags = [f"aesthetic_score_{score}"]
            tag_files(filename=filename, tags=tags)
        else:
            print("Aesthetic Image Scorer: Unable to load Windows tagging script")


class AestheticImageScorer(scripts.Script):
    def title(self):
        return "Aesthetic Image Scorer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p):
        pass


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_before_image_saved)
script_callbacks.on_image_saved(on_image_saved)
