import os
import time
import torch
from PIL import Image
from modules.shared import log
from modules.errors import display

from modules.control.proc.hed import HEDdetector
from modules.control.proc.canny import CannyDetector
from modules.control.proc.edge import EdgeDetector
from modules.control.proc.lineart import LineartDetector
from modules.control.proc.lineart_anime import LineartAnimeDetector
from modules.control.proc.pidi import PidiNetDetector
from modules.control.proc.mediapipe_face import MediapipeFaceDetector
from modules.control.proc.shuffle import ContentShuffleDetector

from modules.control.proc.leres import LeresDetector
from modules.control.proc.midas import MidasDetector
from modules.control.proc.mlsd import MLSDdetector
from modules.control.proc.normalbae import NormalBaeDetector
from modules.control.proc.openpose import OpenposeDetector
from modules.control.proc.dwpose import DWposeDetector
from modules.control.proc.segment_anything import SamDetector
from modules.control.proc.zoe import ZoeDetector


models = {}
cache_dir = 'models/control/processors'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
config = {
    # pose models
    'OpenPose': {'class': OpenposeDetector, 'checkpoint': True, 'params': {'include_body': True, 'include_hand': False, 'include_face': False}},
    'DWPose': {'class': DWposeDetector, 'checkpoint': False, 'model': 'Tiny', 'params': {'min_confidence': 0.3}},
    'MediaPipe Face': {'class': MediapipeFaceDetector, 'checkpoint': False, 'params': {'max_faces': 1, 'min_confidence': 0.5}},
    # outline models
    'Canny': {'class': CannyDetector, 'checkpoint': False, 'params': {'low_threshold': 100, 'high_threshold': 200}},
    'Edge': {'class': EdgeDetector, 'checkpoint': False, 'params': {'pf': True, 'mode': 'edge'}},
    'LineArt Realistic': {'class': LineartDetector, 'checkpoint': True, 'params': {'coarse': False}},
    'LineArt Anime': {'class': LineartAnimeDetector, 'checkpoint': True, 'params': {}},
    'HED': {'class': HEDdetector, 'checkpoint': True, 'params': {'scribble': False, 'safe': False}},
    'PidiNet': {'class': PidiNetDetector, 'checkpoint': True, 'params': {'scribble': False, 'safe': False, 'apply_filter': False}},
    # depth models
    'Midas Depth Hybrid': {'class': MidasDetector, 'checkpoint': True, 'params': {'bg_th': 0.1, 'depth_and_normal': False}},
    'Leres Depth': {'class': LeresDetector, 'checkpoint': True, 'params': {'boost': False, 'thr_a':0, 'thr_b':0}},
    'Zoe Depth': {'class': ZoeDetector, 'checkpoint': True, 'params': {'gamma_corrected': False}, 'load_config': {'pretrained_model_or_path': 'halffried/gyre_zoedepth', 'filename': 'ZoeD_M12_N.safetensors', 'model_type': "zoedepth"}},
    'Normal Bae': {'class': NormalBaeDetector, 'checkpoint': True, 'params': {}},
    # segmentation models
    'SegmentAnything': {'class': SamDetector, 'checkpoint': True, 'model': 'Base', 'params': {}},
    # other models
    'MLSD': {'class': MLSDdetector, 'checkpoint': True, 'params': {'thr_v': 0.1, 'thr_d': 0.1}},
    'Shuffle': {'class': ContentShuffleDetector, 'checkpoint': False, 'params': {}},
    # 'Midas Depth Large': {'class': MidasDetector, 'checkpoint': True, 'params': {'bg_th': 0.1, 'depth_and_normal': False}, 'load_config': {'pretrained_model_or_path': 'Intel/dpt-large', 'model_type': "dpt_large", 'filename': ''}},
    # 'Zoe Depth Zoe': {'class': ZoeDetector, 'checkpoint': True, 'params': {}},
    # 'Zoe Depth NK': {'class': ZoeDetector, 'checkpoint': True, 'params': {}, 'load_config': {'pretrained_model_or_path': 'halffried/gyre_zoedepth', 'filename': 'ZoeD_M12_NK.safetensors', 'model_type': "zoedepth_nk"}},
}


def list_models(refresh=False):
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = ['None'] + list(config)
    debug(f'Control list processors: path={cache_dir} models={models}')
    return models


def update_settings(*settings):
    debug(f'Control settings: {settings}')
    def update(what, val):
        processor_id = what[0]
        if len(what) == 2 and config[processor_id][what[1]] != val:
            config[processor_id][what[1]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')
        elif len(what) == 3 and config[processor_id][what[1]][what[2]] != val:
            config[processor_id][what[1]][what[2]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')
        elif len(what) == 4 and config[processor_id][what[1]][what[2]][what[3]] != val:
            config[processor_id][what[1]][what[2]][what[3]] = val
            config[processor_id]['dirty'] = True
            log.debug(f'Control settings: id="{processor_id}" {what[-1]}={val}')

    update(['HED', 'params', 'scribble'], settings[0])
    update(['Midas Depth Hybrid', 'params', 'bg_th'], settings[1])
    update(['Midas Depth Hybrid', 'params', 'depth_and_normal'], settings[2])
    update(['MLSD', 'params', 'thr_v'], settings[3])
    update(['MLSD', 'params', 'thr_d'], settings[4])
    update(['OpenPose', 'params', 'include_body'], settings[5])
    update(['OpenPose', 'params', 'include_hand'], settings[6])
    update(['OpenPose', 'params', 'include_face'], settings[7])
    update(['PidiNet', 'params', 'scribble'], settings[8])
    update(['PidiNet', 'params', 'apply_filter'], settings[9])
    update(['LineArt Realistic', 'params', 'coarse'], settings[10])
    update(['Leres Depth', 'params', 'boost'], settings[11])
    update(['Leres Depth', 'params', 'thr_a'], settings[12])
    update(['Leres Depth', 'params', 'thr_b'], settings[13])
    update(['MediaPipe Face', 'params', 'max_faces'], settings[14])
    update(['MediaPipe Face', 'params', 'min_confidence'], settings[15])
    update(['Canny', 'params', 'low_threshold'], settings[16])
    update(['Canny', 'params', 'high_threshold'], settings[17])
    update(['DWPose', 'model'], settings[18])
    update(['DWPose', 'params', 'min_confidence'], settings[19])
    update(['SegmentAnything', 'model'], settings[20])
    update(['Edge', 'params', 'pf'], settings[21])
    update(['Edge', 'params', 'mode'], settings[22])
    update(['Zoe Depth', 'params', 'gamma_corrected'], settings[23])


class Processor():
    def __init__(self, processor_id: str = None, resize = True, load_config = None):
        self.model = None
        self.resize = resize
        self.processor_id = processor_id
        self.override = None # override input image
        self.load_config = { 'cache_dir': cache_dir }
        from_config = config.get(processor_id, {}).get('load_config', None)
        if load_config is not None:
            for k, v in load_config.items():
                self.load_config[k] = v
        if from_config is not None:
            for k, v in from_config.items():
                self.load_config[k] = v
        if processor_id is not None:
            self.load()

    def reset(self):
        if self.model is not None:
            log.debug(f'Control processor unloaded: id="{self.processor_id}"')
        self.model = None
        self.processor_id = None
        self.override = None

    def load(self, processor_id: str = None) -> str:
        try:
            t0 = time.time()
            processor_id = processor_id or self.processor_id
            if processor_id is None or processor_id == 'None':
                self.reset()
                return ''
            from_config = config.get(processor_id, {}).get('load_config', None)
            if from_config is not None:
                for k, v in from_config.items():
                    self.load_config[k] = v
            cls = config[processor_id]['class']
            log.debug(f'Control processor loading: id="{processor_id}" class={cls.__name__}')
            debug(f'Control processor config={self.load_config}')
            if 'DWPose' in processor_id:
                det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
                if 'Tiny' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-tt_ucoco.pth'
                elif 'Medium' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-mm_ucoco.pth'
                elif 'Large' == config['DWPose']['model']:
                    pose_config = 'config/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
                    pose_ckpt = 'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth'
                else:
                    log.error(f'Control processor load failed: id="{processor_id}" error=unknown model type')
                    return f'Processor failed to load: {processor_id}'
                self.model = cls(det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device="cpu")
            elif 'SegmentAnything' in processor_id:
                if 'Base' == config['SegmentAnything']['model']:
                    self.model = cls.from_pretrained(model_path = 'segments-arnaud/sam_vit_b', filename='sam_vit_b_01ec64.pth', model_type='vit_b', **self.load_config)
                elif 'Large' == config['SegmentAnything']['model']:
                    self.model = cls.from_pretrained(model_path = 'segments-arnaud/sam_vit_l', filename='sam_vit_l_0b3195.pth', model_type='vit_l', **self.load_config)
                else:
                    log.error(f'Control processor load failed: id="{processor_id}" error=unknown model type')
                    return f'Processor failed to load: {processor_id}'
            elif config[processor_id].get('load_config', None) is not None:
                self.model = cls.from_pretrained(**self.load_config)
            elif config[processor_id]['checkpoint']:
                self.model = cls.from_pretrained("lllyasviel/Annotators", **self.load_config)
            else:
                self.model = cls() # class instance only
            t1 = time.time()
            self.processor_id = processor_id
            log.debug(f'Control processor loaded: id="{processor_id}" class={self.model.__class__.__name__} time={t1-t0:.2f}')
            return f'Processor loaded: {processor_id}'
        except Exception as e:
            log.error(f'Control processor load failed: id="{processor_id}" error={e}')
            display(e, 'Control processor load')
            return f'Processor load filed: {processor_id}'

    def __call__(self, image_input: Image):
        if self.override is not None:
            image_input = self.override
        image_process = image_input
        if image_input is None:
            log.error('Control processor: no input')
            return image_process
        if self.model is None:
            # log.error('Control processor: model not loaded')
            return image_process
        if config[self.processor_id].get('dirty', False):
            processor_id = self.processor_id
            config[processor_id].pop('dirty')
            self.reset()
            self.load(processor_id)
        try:
            t0 = time.time()
            kwargs = config.get(self.processor_id, {}).get('params', None)
            if self.resize:
                orig_size = image_input.size
                image_resized = image_input.resize((512, 512))
            else:
                image_resized = image_input
            with torch.no_grad():
                image_process = self.model(image_resized, **kwargs)
            if self.resize:
                image_process = image_process.resize(orig_size, Image.Resampling.LANCZOS)
            t1 = time.time()
            log.debug(f'Control processor: id="{self.processor_id}" args={kwargs} time={t1-t0:.2f}')
        except Exception as e:
            log.error(f'Control processor failed: id="{self.processor_id}" error={e}')
            display(e, 'Control processor')
        return image_process

    def preview(self, image_input: Image):
        return self.__call__(image_input)
