import torchvision # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
from einops import rearrange
from .densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr, densepose_chart_predictor_output_to_result_with_confidences
from modules import devices
from annotator.annotator_path import models_path
import os

N_PART_LABELS = 24
result_visualizer = DensePoseMaskedColormapResultsVisualizer(
    alpha=1,
    data_extractor=_extract_i_from_iuvarr,
    segm_extractor=_extract_i_from_iuvarr,
    val_scale = 255.0 / N_PART_LABELS
)
remote_torchscript_path = "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image/resolve/main/densepose_r50_fpn_dl.torchscript"
torchscript_model = None
model_dir = os.path.join(models_path, "densepose")

def apply_densepose(input_image, cmap="viridis"):
    global torchscript_model
    if torchscript_model is None:
        model_path = os.path.join(model_dir, "densepose_r50_fpn_dl.torchscript")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_torchscript_path, model_dir=model_dir)
        torchscript_model = torch.jit.load(model_path, map_location="cpu").to(devices.get_device_for("controlnet")).eval()
    H, W  = input_image.shape[:2]

    hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
    hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])
    input_image = rearrange(torch.from_numpy(input_image).to(devices.get_device_for("controlnet")), 'h w c -> c h w')
    pred_boxes, corase_segm, fine_segm, u, v = torchscript_model(input_image)

    extractor = densepose_chart_predictor_output_to_result_with_confidences
    densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

    if cmap=="viridis":
        result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
        hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
        hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
        hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
        hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
        hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
    else:
        result_visualizer.mask_visualizer.cmap = cv2.COLORMAP_PARULA
        hint_image = result_visualizer.visualize(hint_image_canvas, densepose_results)
        hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

    return hint_image

def unload_model():
    global torchscript_model
    if torchscript_model is not None:
        torchscript_model.cpu()
