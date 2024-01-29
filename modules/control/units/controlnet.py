import os
import time
from typing import Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
from modules.control.units import detect
from modules.shared import log, opts, listdir
from modules import errors


what = 'ControlNet'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
predefined_sd15 = {
    'Canny': "lllyasviel/control_v11p_sd15_canny",
    'Depth': "lllyasviel/control_v11f1p_sd15_depth",
    'HED': "lllyasviel/sd-controlnet-hed",
    'IP2P': "lllyasviel/control_v11e_sd15_ip2p",
    'LineArt': "lllyasviel/control_v11p_sd15_lineart",
    'LineArt Anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'MLDS': "lllyasviel/control_v11p_sd15_mlsd",
    'NormalBae': "lllyasviel/control_v11p_sd15_normalbae",
    'OpenPose': "lllyasviel/control_v11p_sd15_openpose",
    'Scribble': "lllyasviel/control_v11p_sd15_scribble",
    'Segment': "lllyasviel/control_v11p_sd15_seg",
    'Shuffle': "lllyasviel/control_v11e_sd15_shuffle",
    'SoftEdge': "lllyasviel/control_v11p_sd15_softedge",
    'Tile': "lllyasviel/control_v11f1e_sd15_tile",
    'Depth Anything': 'vladmandic/depth-anything',
    'Canny FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_canny.safetensors',
    'Inpaint FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_inpaint.safetensors',
    'LineArt Anime FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_animeline.safetensors',
    'LineArt FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_lineart.safetensors',
    'MLSD FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_mlsd.safetensors',
    'NormalBae FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_normal.safetensors',
    'OpenPose FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_openpose.safetensors',
    'Pix2Pix FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_pix2pix.safetensors',
    'Scribble FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_scribble.safetensors',
    'Segment FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_seg.safetensors',
    'Shuffle FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_shuffle.safetensors',
    'SoftEdge FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_softedge.safetensors',
    'Tile FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_tileE.safetensors',
    'CiaraRowles TemporalNet': "CiaraRowles/TemporalNet",
    'Ciaochaos Recolor': 'ioclab/control_v1p_sd15_brightness',
    'Ciaochaos Illumination': 'ioclab/control_v1u_sd15_illumination/illumination20000.safetensors',
}
predefined_sdxl = {
    'Canny Small XL': 'diffusers/controlnet-canny-sdxl-1.0-small',
    'Canny Mid XL': 'diffusers/controlnet-canny-sdxl-1.0-mid',
    'Canny XL': 'diffusers/controlnet-canny-sdxl-1.0',
    'Depth Zoe XL': 'diffusers/controlnet-zoe-depth-sdxl-1.0',
    'Depth Mid XL': 'diffusers/controlnet-depth-sdxl-1.0-mid',
    'OpenPose XL': 'thibaud/controlnet-openpose-sdxl-1.0',
    # 'StabilityAI Canny R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-canny-rank128.safetensors',
    # 'StabilityAI Depth R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-depth-rank128.safetensors',
    # 'StabilityAI Recolor R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors',
    # 'StabilityAI Sketch R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors',
    # 'StabilityAI Canny R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-canny-rank256.safetensors',
    # 'StabilityAI Depth R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-depth-rank256.safetensors',
    # 'StabilityAI Recolor R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors',
    # 'StabilityAI Sketch R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors',
}
models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
cache_dir = 'models/control/controlnet'


def find_models():
    path = os.path.join(opts.control_dir, 'controlnet')
    files = listdir(path)
    files = [f for f in files if f.endswith('.safetensors')]
    downloaded_models = {}
    for f in files:
        basename = os.path.splitext(os.path.relpath(f, path))[0]
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
        models = ['None'] + list(predefined_sdxl) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd':
        models = ['None'] + list(predefined_sd15) + sorted(find_models())
    else:
        log.warning(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(predefined_sd15) + sorted(predefined_sdxl) + sorted(find_models())
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class ControlNet():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: ControlNetModel = None
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
            debug(f'Control {what} model unloaded')
        self.model = None
        self.model_id = None

    def load_safetensors(self, model_path):
        name = os.path.splitext(model_path)[0]
        config_path = None
        if not os.path.exists(model_path):
            import huggingface_hub as hf
            parts = model_path.split('/')
            repo_id = f'{parts[0]}/{parts[1]}'
            filename = os.path.splitext('/'.join(parts[2:]))[0]
            model_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.safetensors', cache_dir=cache_dir)
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.yaml', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.json', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
        elif os.path.exists(name + '.yaml'):
            config_path = f'{name}.yaml'
        elif os.path.exists(name + '.json'):
            config_path = f'{name}.json'
        if config_path is not None:
            self.load_config['original_config_file '] = config_path
        self.model = ControlNetModel.from_single_file(model_path, **self.load_config)

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
            log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}"')
            if model_path.endswith('.safetensors'):
                self.load_safetensors(model_path)
            else:
                self.model = ControlNetModel.from_pretrained(model_path, **self.load_config)
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


class ControlNetPipeline():
    def __init__(self, controlnet: Union[ControlNetModel, list[ControlNetModel]], pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline], dtype = None):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None
        if pipeline is None:
            log.error('Control model pipeline: model not loaded')
            return
        elif detect.is_sdxl(pipeline):
            self.pipeline = StableDiffusionXLControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                controlnet=controlnet, # can be a list
            ).to(pipeline.device)
        elif detect.is_sd15(pipeline):
            self.pipeline = StableDiffusionControlNetPipeline(
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
