from PIL import Image
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from modules import devices
from modules.postprocess.scunet_model_arch import SCUNet as net
from modules.shared import opts, log, console, device
from modules.upscaler import Upscaler, compile_upscaler


class UpscalerSCUNet(Upscaler):
    def __init__(self, dirname):
        self.name = "SCUNet"
        self.user_path = dirname
        super().__init__()
        self.scalers = self.find_scalers()
        self.models = {}

    def load_model(self, path: str):
        info = self.find_model(path)
        if info is None:
            return
        if self.models.get(info.local_data_path, None) is not None:
            log.debug(f"Upscaler cached: type={self.name} model={info.local_data_path}")
            model=self.models[info.local_data_path]
        else:
            model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
            model.load_state_dict(torch.load(info.local_data_path), strict=True)
            model.eval()
            log.info(f"Upscaler loaded: type={self.name} model={info.local_data_path}")
            for _, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)
            model = compile_upscaler(model, name=self.name)
            self.models[info.local_data_path] = model
        return model

    @staticmethod
    @torch.no_grad()
    def tiled_inference(img, model):
        # test the image tile by tile
        h, w = img.shape[2:]
        tile = opts.upscaler_tile_size
        tile_overlap = opts.upscaler_tile_overlap
        if tile == 0:
            return model(img)
        assert tile % 8 == 0, "tile size should be a multiple of window_size"
        sf = 1
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(1, 3, h * sf, w * sf, dtype=img.dtype, device=device)
        W = torch.zeros_like(E, dtype=devices.dtype, device=device)
        with Progress(TextColumn('[cyan]{task.description}'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), TimeElapsedColumn(), console=console) as progress:
            task = progress.add_task(description="Upscaling", total=len(h_idx_list) * len(w_idx_list))
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch)
                    W[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch_mask)
                    progress.update(task, advance=1, description="Upscaling")
        output = E.div_(W)
        return output

    def do_upscale(self, img: Image.Image, selected_file):
        devices.torch_gc()
        model = self.load_model(selected_file)
        if model is None:
            return img
        tile = opts.upscaler_tile_size
        h, w = img.height, img.width
        np_img = np.array(img)
        np_img = np_img[:, :, ::-1]  # RGB to BGR
        np_img = np_img.transpose((2, 0, 1)) / 255  # HWC to CHW
        torch_img = torch.from_numpy(np_img).float().unsqueeze(0).to(device)  # type: ignore
        if tile > h or tile > w:
            _img = torch.zeros(1, 3, max(h, tile), max(w, tile), dtype=torch_img.dtype, device=torch_img.device)
            _img[:, :, :h, :w] = torch_img # pad image
            torch_img = _img
        torch_output = self.tiled_inference(torch_img, model).squeeze(0)
        torch_output = torch_output[:, :h * 1, :w * 1] # remove padding, if any
        np_output: np.ndarray = torch_output.float().cpu().clamp_(0, 1).numpy()
        del torch_img, torch_output
        devices.torch_gc()
        output = np_output.transpose((1, 2, 0))  # CHW to HWC
        output = output[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray((output * 255).astype(np.uint8))
        if opts.upscaler_unload and selected_file in self.models:
            del self.models[selected_file]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_file}")
            devices.torch_gc(force=True)
        return img
