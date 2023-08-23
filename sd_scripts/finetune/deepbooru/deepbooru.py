import os
import re

import torch
from PIL import Image
import numpy as np

# from modules import modelloader, paths, deepbooru_model, devices, images, shared
from sd_scripts.finetune.deepbooru import deepbooru_model
from sd_scripts.super_upscaler.super_upscaler import upscaler

re_special = re.compile(r'([\\()])')

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 

def resize_image(resize_mode, im, width, height, upscaler_name=None, models_path=""):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    upscaler_name = upscaler_name
    LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    def resize(im, w, h):
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            # upscalers = [x for x in shared.sd_upscalers if x.name == upscaler_name]
            # if len(upscalers) == 0:
            #     upscaler = shared.sd_upscalers[0]
            #     print(f"could not find upscaler named {upscaler_name or '<empty string>'}, using {upscaler.name} as a fallback")
            # else:
            #     upscaler = upscalers[0]

            # im = upscaler.scaler.upscale(im, scale, upscaler.data_path)
            im = upscaler(im, upscale_by=scale, style_type=1, upscaler_2_visibility=0.3, swap=True,models_path=models_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res

class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self,models_path):
        if self.model is not None:
            return

        # files = modelloader.load_models(
        #     model_path=os.path.join(paths.models_path, "torch_deepdanbooru"),
        #     model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
        #     ext_filter=[".pt"],
        #     download_name='model-resnet_custom_v3.pt',
        # )
        model_path=os.path.join(models_path, "torch_deepdanbooru/model-resnet_custom_v3.pt")
        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.model.eval()
        self.model.to("cuda")

    def start(self,models_path):
        self.load(models_path)
        self.model.to("cuda")

    def stop(self):
        # if not shared.opts.interrogate_keep_models_in_memory:
        self.model.to("cpu")
        torch_gc()

    def tag(self, pil_image,models_path):
        self.start(models_path)
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(self, pil_image, force_disable_ranks=False,interrogate_deepbooru_score_threshold=0.5,deepbooru_use_spaces=False,
            deepbooru_escape=False,deepbooru_sort_alpha=False,interrogate_return_ranks=False,deepbooru_filter_tags="",addtional_tags=""):
        threshold = interrogate_deepbooru_score_threshold
        use_spaces = deepbooru_use_spaces
        use_escape = deepbooru_escape
        alpha_sort = deepbooru_sort_alpha
        include_ranks = interrogate_return_ranks and not force_disable_ranks

        pic = resize_image(2, pil_image.convert("RGB"), 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).to("cuda")
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        filtertags = set([x.strip().replace(' ', '_') for x in deepbooru_filter_tags.split(",")])

        for tag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)


model = DeepDanbooru()
