import os
import time
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, T2IAdapter, StableDiffusionAdapterPipeline, StableDiffusionXLAdapterPipeline
from modules.shared import log
from modules import errors


what = 'T2I-Adapter'
debug = log.debug if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
predefined_sd15 = {
    'Canny': 'TencentARC/t2iadapter_canny_sd15v2',
    'Depth': 'TencentARC/t2iadapter_depth_sd15v2',
    'Depth Zoe': 'TencentARC/t2iadapter_zoedepth_sd15v1',
    'OpenPose': 'TencentARC/t2iadapter_openpose_sd14v1',
    'Sketch': 'TencentARC/t2iadapter_sketch_sd15v2',
}
predefined_sdxl = {
    'Canny XL': 'TencentARC/t2i-adapter-canny-sdxl-1.0',
    'Depth Zoe XL': 'TencentARC/t2i-adapter-depth-zoe-sdxl-1.0',
    'Depth Midas XL': 'TencentARC/t2i-adapter-depth-midas-sdxl-1.0',
    'LineArt XL': 'TencentARC/t2i-adapter-lineart-sdxl-1.0',
    'OpenPose XL': 'TencentARC/t2i-adapter-openpose-sdxl-1.0',
    'Sketch XL': 'TencentARC/t2i-adapter-sketch-sdxl-1.0',
}
models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
cache_dir = 'models/control/adapters'


def list_models(refresh=False):
    import modules.shared
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = {}
    if modules.shared.sd_model_type == 'none':
        models = ['None']
    if modules.shared.sd_model_type == 'sdxl':
        models = ['None'] + sorted(predefined_sdxl)
    if modules.shared.sd_model_type == 'sd':
        models = ['None'] + sorted(predefined_sd15)
    else:
        models = ['None'] + sorted(list(predefined_sd15) + list(predefined_sdxl))
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class AdapterModel(T2IAdapter):
    pass


class Adapter():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: AdapterModel = None
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

    def load(self, model_id: str = None):
        try:
            t0 = time.time()
            model_id = model_id or self.model_id
            if model_id is None or model_id == 'None':
                self.reset()
                return
            model_path = all_models[model_id]
            if model_path is None:
                log.error(f'Control {what} model load failed: id="{model_id}" error=unknown model id')
                return
            log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}"')
            self.model = T2IAdapter.from_pretrained(model_path, **self.load_config)
            if self.device is not None:
                self.model.to(self.device)
            if self.dtype is not None:
                self.model.to(self.dtype)
            t1 = time.time()
            self.model_id = model_id
            log.debug(f'Control {what} loaded: id="{model_id}" path="{model_path}" time={t1-t0:.2f}')
        except Exception as e:
            log.error(f'Control {what} model load failed: id="{model_id}" error={e}')
            errors.display(e, f'Control {what} load')


class AdapterPipeline():
    def __init__(self, adapter: T2IAdapter | list[T2IAdapter], pipeline: StableDiffusionXLPipeline | StableDiffusionPipeline, dtype = None):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None
        if pipeline is None:
            log.error(f'Control {what} pipeline: model not loaded')
            return
        if isinstance(pipeline, StableDiffusionXLPipeline):
            self.pipeline = StableDiffusionXLAdapterPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                adapter=adapter, # can be a list
            ).to(pipeline.device)
        elif isinstance(pipeline, StableDiffusionPipeline):
            self.pipeline = StableDiffusionAdapterPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                requires_safety_checker=False,
                safety_checker=None,
                feature_extractor=None,
                adapter=adapter, # can be a list
            ).to(pipeline.device)
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return
        if dtype is not None and self.pipeline is not None:
            self.pipeline.dtype = dtype
        t1 = time.time()
        if self.pipeline is not None:
            log.debug(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')
        else:
            log.error(f'Control {what} pipeline: not initialized')


    def restore(self):
        self.pipeline = None
        return self.orig_pipeline
