import os
import time
from typing import Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from modules.shared import log, opts, listdir
from modules import errors
from modules.control.units.xs_model import ControlNetXSModel
from modules.control.units.xs_pipe import StableDiffusionControlNetXSPipeline, StableDiffusionXLControlNetXSPipeline
from modules.control.units import detect


what = 'ControlNet-XS'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
predefined_sd15 = {
}
predefined_sdxl = {
    'Canny': 'UmerHA/ConrolNetXS-SDXL-canny',
    'Depth': 'UmerHA/ConrolNetXS-SDXL-depth',
}
models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
cache_dir = 'models/control/xs'


def find_models():
    path = os.path.join(opts.control_dir, 'xs')
    files = listdir(path)
    files = [f for f in files if f.endswith('.safetensors')]
    downloaded_models = {}
    for f in files:
        basename = os.path.splitext(f)[0]
        downloaded_models[basename] = os.path.join(path, f)
    all_models.update(downloaded_models)
    return downloaded_models


def list_models(refresh=False):
    global models # pylint: disable=global-statement
    import modules.shared
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
        log.error(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(predefined_sd15) + sorted(predefined_sdxl) + sorted(find_models())
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class ControlNetXS():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: ControlNetXSModel = None
        self.model_id: str = model_id
        self.device = device
        self.dtype = dtype
        self.load_config = { 'cache_dir': cache_dir, 'learn_embedding': True }
        if load_config is not None:
            self.load_config.update(load_config)
        if model_id is not None:
            self.load()

    def reset(self):
        if self.model is not None:
            log.debug(f'Control {what} model unloaded')
        self.model = None
        self.model_id = None

    def load(self, model_id: str = None, time_embedding_mix: float = 0.0) -> str:
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
            self.load_config['time_embedding_mix'] = time_embedding_mix
            log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}" {self.load_config}')
            if model_path.endswith('.safetensors'):
                self.model = ControlNetXSModel.from_single_file(model_path, **self.load_config)
            else:
                self.model = ControlNetXSModel.from_pretrained(model_path, **self.load_config)
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


class ControlNetXSPipeline():
    def __init__(self, controlnet: Union[ControlNetXSModel, list[ControlNetXSModel]], pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline], dtype = None):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None
        if pipeline is None:
            log.error(f'Control {what} pipeline: model not loaded')
            return
        if detect.is_sdxl(pipeline):
            self.pipeline = StableDiffusionXLControlNetXSPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                # feature_extractor=getattr(pipeline, 'feature_extractor', None),
                controlnet=controlnet, # can be a list
            ).to(pipeline.device)
        elif detect.is_sd15(pipeline):
            self.pipeline = StableDiffusionControlNetXSPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
                controlnet=controlnet, # can be a list
            ).to(pipeline.device)
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return
        if dtype is not None and self.pipeline is not None:
            self.pipeline = self.pipeline.to(dtype)
        t1 = time.time()
        if self.pipeline is not None:
            log.debug(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')
        else:
            log.error(f'Control {what} pipeline: not initialized')

    def restore(self):
        self.pipeline = None
        return self.orig_pipeline
