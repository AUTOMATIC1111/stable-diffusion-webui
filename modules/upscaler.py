import os
import copy
import time
import logging
from abc import abstractmethod
from PIL import Image
from modules import devices, modelloader, shared
from installer import setup_logging


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.Resampling.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.Resampling.NEAREST)
models = None

class Upscaler:
    name = None
    folder = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers = []
    tile = True

    def __init__(self, create_dirs=True):
        global models # pylint: disable=global-statement
        if models is None:
            models = shared.readfile('html/upscalers.json')
        self.mod_pad_h = None
        self.tile_size = shared.opts.upscaler_tile_size
        self.tile_pad = shared.opts.upscaler_tile_overlap
        self.device = shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None
        if self.user_path is not None and len(self.user_path) > 0 and not os.path.exists(self.user_path):
            shared.log.info(f'Upscaler create: folder="{self.user_path}"')
        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)
        try:
            import cv2  # pylint: disable=unused-import
            self.can_tile = True
        except Exception:
            pass

    def find_folder(self, folder, scalers, loaded):
        for fn in os.listdir(folder): # from folder
            file_name = os.path.join(folder, fn)
            if os.path.isdir(file_name):
                self.find_folder(file_name, scalers, loaded)
                continue
            if not file_name.endswith('.pth') and not file_name.endswith('.pt'):
                continue
            if file_name not in loaded:
                model_name = os.path.splitext(fn)[0]
                scaler = UpscalerData(name=f'{self.name} {model_name}', path=file_name, upscaler=self)
                scaler.custom = True
                scalers.append(scaler)
                loaded.append(file_name)
                shared.log.debug(f'Upscaler type={self.name} folder="{folder}" model="{model_name}" path="{file_name}"')

    def find_scalers(self):
        scalers = []
        loaded = []
        for k, v in models.items(): # from config
            if k != self.name:
                continue
            for model in v:
                local_name = os.path.join(self.user_path, modelloader.friendly_fullname(model[1]))
                model_path = local_name if os.path.exists(local_name) else model[1]
                scaler = UpscalerData(name=f'{k} {model[0]}', path=model_path, upscaler=self)
                scalers.append(scaler)
                loaded.append(model_path)
                # shared.log.debug(f'Upscaler type={self.name} folder="{self.user_path}" model="{model[0]}" path="{model_path}"')
        if not os.path.exists(self.user_path):
            return scalers
        self.find_folder(self.user_path, scalers, loaded)
        return scalers

    @abstractmethod
    def do_upscale(self, img: Image, selected_model: str):
        return img

    def upscale(self, img: Image, scale, selected_model: str = None):
        orig_state = copy.deepcopy(shared.state)
        shared.state.begin('upscale')
        self.scale = scale
        dest_w = int(img.width * scale)
        dest_h = int(img.height * scale)
        for _ in range(3):
            shape = (img.width, img.height)
            img = self.do_upscale(img, selected_model)
            if shape == (img.width, img.height):
                break
            if img.width >= dest_w and img.height >= dest_h:
                break
        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)
        shared.state.end()
        shared.state = orig_state
        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def find_models(self, ext_filter=None) -> list: # pylint: disable=unused-argument
        return modelloader.load_models(model_path=self.model_path, model_url=self.model_url, command_path=self.user_path)

    def update_status(self, prompt):
        shared.log.info(f'Upscaler: type={self.name} model="{prompt}"')

    def find_model(self, path):
        info = None
        for scaler in self.scalers:
            if scaler.data_path == path:
                info = scaler
                break
        if info is None:
            shared.log.error(f'Upscaler cannot match model: type={self.name} model="{path}"')
            return None
        if info.local_data_path.startswith("http"):
            from modules.modelloader import load_file_from_url
            info.local_data_path = load_file_from_url(url=info.data_path, model_dir=self.model_download_path, progress=True)
        if not os.path.isfile(info.local_data_path):
            shared.log.error(f'Upscaler cannot find model: type={self.name} model="{info.local_data_path}"')
            return None
        return info


class UpscalerData:
    custom: bool = False
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(self, name: str, path: str, upscaler: Upscaler = None, scale: int = 4, model=None):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model


class UpscalerNone(Upscaler):
    name = "None"
    scalers = []

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):
    scalers = []

    def do_upscale(self, img, selected_model=None):
        return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=NEAREST)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]

def compile_upscaler(model):
    try:
        if shared.opts.ipex_optimize and "Upscaler" in shared.opts.ipex_optimize:
            t0 = time.time()
            import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
            model.eval()
            model.training = False
            model = ipex.optimize(model, dtype=devices.dtype, inplace=True, weights_prepack=False) # pylint: disable=attribute-defined-outside-init
            t1 = time.time()
            shared.log.info(f"Upscaler IPEX Optimize: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Upscaler IPEX Optimize: error: {e}")

    try:
        if "Upscaler" in shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
            import torch._dynamo # pylint: disable=unused-import,redefined-outer-name
            torch._dynamo.reset() # pylint: disable=protected-access
            shared.log.debug(f"Upscaler compile available backends: {torch._dynamo.list_backends()}") # pylint: disable=protected-access

            if shared.opts.cuda_compile_backend == "openvino_fx":
                from modules.intel.openvino import openvino_fx # pylint: disable=unused-import
                torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True # pylint: disable=protected-access

            log_level = logging.WARNING if shared.opts.cuda_compile_verbose else logging.CRITICAL # pylint: disable=protected-access
            if hasattr(torch, '_logging'):
                torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level) # pylint: disable=protected-access
            torch._dynamo.config.verbose = shared.opts.cuda_compile_verbose # pylint: disable=protected-access
            torch._dynamo.config.suppress_errors = shared.opts.cuda_compile_errors # pylint: disable=protected-access

            try:
                torch._inductor.config.conv_1x1_as_mm = True # pylint: disable=protected-access
                torch._inductor.config.coordinate_descent_tuning = True # pylint: disable=protected-access
                torch._inductor.config.epilogue_fusion = False # pylint: disable=protected-access
                torch._inductor.config.coordinate_descent_check_all_directions = True # pylint: disable=protected-access
                torch._inductor.config.use_mixed_mm = True # pylint: disable=protected-access
                # torch._inductor.config.force_fuse_int_mm_with_mul = True # pylint: disable=protected-access
            except Exception as e:
                shared.log.error(f"Torch inductor config error: {e}")

            t0 = time.time()

            model = torch.compile(model, mode=shared.opts.cuda_compile_mode, backend=shared.opts.cuda_compile_backend, fullgraph=shared.opts.cuda_compile_fullgraph) # pylint: disable=attribute-defined-outside-init

            setup_logging() # compile messes with logging so reset is needed

            t1 = time.time()
            shared.log.info(f"Upscaler compile: time={t1-t0:.2f}")
    except Exception as e:
        shared.log.warning(f"Upscaler compile error: {e}")
    return model
