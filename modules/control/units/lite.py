import os
import time
from typing import Union
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from modules.shared import log, opts, listdir
from modules import errors
from modules.control.units.lite_model import ControlNetLLLite


what = 'ControlLLLite'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
predefined_sd15 = {
}
predefined_sdxl = {
    'Canny XL': 'kohya-ss/controlnet-lllite/controllllite_v01032064e_sdxl_canny',
    'Canny anime XL': 'kohya-ss/controlnet-lllite/controllllite_v01032064e_sdxl_canny_anime',
    'Depth anime XL': 'kohya-ss/controlnet-lllite/controllllite_v01008016e_sdxl_depth_anime',
    'Blur anime XL': 'kohya-ss/controlnet-lllite/controllllite_v01016032e_sdxl_blur_anime_beta',
    'Pose anime XL': 'kohya-ss/controlnet-lllite/controllllite_v01032064e_sdxl_pose_anime',
    'Replicate anime XL': 'kohya-ss/controlnet-lllite/controllllite_v01032064e_sdxl_replicate_anime_v2',
}
models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
cache_dir = 'models/control/lite'


def find_models():
    path = os.path.join(opts.control_dir, 'lite')
    files = listdir(path)
    files = [f for f in files if f.endswith('.safetensors')]
    downloaded_models = {}
    for f in files:
        basename = os.path.splitext(f)[0]
        downloaded_models[basename] = os.path.join(path, f)
    all_models.update(downloaded_models)
    return downloaded_models


def list_models(refresh=False):
    import modules.shared
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = {}
    if modules.shared.sd_model_type == 'none':
        models = ['None']
    elif modules.shared.sd_model_type == 'sdxl':
        models = ['None'] + sorted(predefined_sdxl) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd':
        models = ['None'] + sorted(predefined_sd15) + sorted(find_models())
    else:
        log.warning(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(predefined_sd15) + sorted(predefined_sdxl) + sorted(find_models())
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class ControlLLLite():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: ControlNetLLLite = None
        self.model_id: str = model_id
        self.device = device
        self.dtype = dtype
        self.load_config = { 'cache_dir': cache_dir }
        if load_config is not None:
            self.load_config.update(load_config)
        if model_id is not None:
            self.load()

    def reset(self):
        if self.model is not None:
            log.debug(f'Control {what} model unloaded')
        self.model = None
        self.model_id = None

    def load(self, model_id: str = None) -> str:
        try:
            t0 = time.time()
            model_id = model_id or self.model_id
            if model_id is None or model_id == 'None':
                self.reset()
                return
            model_path = all_models[model_id]
            if model_path == '':
                return
            if model_path is None:
                log.error(f'Control {what} model load failed: id="{model_id}" error=unknown model id')
                return
            log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}" {self.load_config}')
            if model_path.endswith('.safetensors'):
                self.model = ControlNetLLLite(model_path)
            else:
                import huggingface_hub as hf
                folder, filename = os.path.split(model_path)
                model_path = hf.hf_hub_download(repo_id=folder, filename=f'{filename}.safetensors', cache_dir=cache_dir)
                self.model = ControlNetLLLite(model_path)
            if self.device is not None:
                self.model.to(self.device)
            if self.dtype is not None:
                self.model.to(self.dtype)
            t1 = time.time()
            self.model_id = model_id
            log.debug(f'Control {what} model loaded: id="{model_id}" path="{model_path}" time={t1-t0:.2f}')
            return f'{what} loaded model: {model_id}'
        except Exception as e:
            log.error(f'Control {what} model load failed: id="{model_id}" error={e}')
            errors.display(e, f'Control {what} load')
            return f'{what} failed to load model: {model_id}'


class ControlLLitePipeline():
    def __init__(self, pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline]):
        self.pipeline = pipeline
        self.nets = []

    def apply(self, controlnet: Union[ControlNetLLLite, list[ControlNetLLLite]], image, conditioning):
        if image is None:
            return
        self.nets = [controlnet] if isinstance(controlnet, ControlNetLLLite) else controlnet
        debug(f'Control {what} apply: models={len(self.nets)} image={image} conditioning={conditioning}')
        weight = [conditioning] if isinstance(conditioning, float) else conditioning
        images = [image] if isinstance(image, Image.Image) else image
        images = [i.convert('RGB') for i in images]
        for i, cn in enumerate(self.nets):
            cn.apply(pipe=self.pipeline, cond=np.asarray(images[i % len(images)]), weight=weight[i % len(weight)])

    def restore(self):
        from modules.control.units.lite_model import clear_all_lllite
        clear_all_lllite()
        self.nets = []
