import os
from annotator.annotator_path import models_path
from modules import devices
from annotator.uniformer.inference import init_segmentor, inference_segmentor, show_result_pyplot

try:
    from mmseg.core.evaluation import get_palette
except ImportError:
    from annotator.mmpkg.mmseg.core.evaluation import get_palette

modeldir = os.path.join(models_path, "uniformer")
checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "upernet_global_small.py")
old_modeldir = os.path.dirname(os.path.realpath(__file__))
model = None

def unload_uniformer_model():
    global model
    if model is not None:
        model = model.cpu()

def apply_uniformer(img):
    global model
    if model is None:
        modelpath = os.path.join(modeldir, "upernet_global_small.pth")
        old_modelpath = os.path.join(old_modeldir, "upernet_global_small.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath  
        elif not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(checkpoint_file, model_dir=modeldir)
            
        model = init_segmentor(config_file, modelpath, device=devices.get_device_for("controlnet"))
    model = model.to(devices.get_device_for("controlnet"))
    
    if devices.get_device_for("controlnet").type == 'mps':
        # adaptive_avg_pool2d can fail on MPS, workaround with CPU
        import torch.nn.functional
        
        orig_adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d
        def cpu_if_exception(input, *args, **kwargs):
            try:
                return orig_adaptive_avg_pool2d(input, *args, **kwargs)
            except:
                return orig_adaptive_avg_pool2d(input.cpu(), *args, **kwargs).to(input.device)
        
        try:
            torch.nn.functional.adaptive_avg_pool2d = cpu_if_exception
            result = inference_segmentor(model, img)
        finally:
            torch.nn.functional.adaptive_avg_pool2d = orig_adaptive_avg_pool2d
    else:
        result = inference_segmentor(model, img)
    
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    return res_img
