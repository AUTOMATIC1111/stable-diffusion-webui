import os
import cv2
import numpy as np
import torch
import math
from dataclasses import dataclass
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput

from annotator.util import HWC3
from typing import Callable, Tuple, Union

from modules.safe import Extra
from modules import devices
from scripts.logging import logger


def torch_handler(module: str, name: str):
    """ Allow all torch access. Bypass A1111 safety whitelist. """
    if module == 'torch':
        return getattr(torch, name)
    if module == 'torch._tensor':
        # depth_anything dep.
        return getattr(torch._tensor, name)


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


model_canny = None


def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_canny
    if model_canny is None:
        from annotator.canny import apply_canny
        model_canny = apply_canny
    result = model_canny(img, l, h)
    return remove_pad(result), True


def scribble_thr(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) < 127] = 255
    return remove_pad(result), True


def scribble_xdog(img, res=512, thr_a=32, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(img, dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    return remove_pad(result), True


def tile_resample(img, res=512, thr_a=1.0, **kwargs):
    img = HWC3(img)
    if thr_a < 1.1:
        return img, True
    H, W, C = img.shape
    H = int(float(H) / float(thr_a))
    W = int(float(W) / float(thr_a))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, True


def threshold(img, res=512, thr_a=127, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    result = np.zeros_like(img, dtype=np.uint8)
    result[np.min(img, axis=2) > thr_a] = 255
    return remove_pad(result), True


def identity(img, **kwargs):
    return img, True


def invert(img, res=512, **kwargs):
    return 255 - HWC3(img), True


model_hed = None


def hed(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img)
    return remove_pad(result), True


def hed_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_hed
    if model_hed is None:
        from annotator.hed import apply_hed
        model_hed = apply_hed
    result = model_hed(img, is_safe=True)
    return remove_pad(result), True


def unload_hed():
    global model_hed
    if model_hed is not None:
        from annotator.hed import unload_hed_model
        unload_hed_model()


def scribble_hed(img, res=512, **kwargs):
    result, _ = hed(img, res)
    import cv2
    from annotator.util import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result, True


model_mediapipe_face = None


def mediapipe_face(img, res=512, thr_a: int = 10, thr_b: float = 0.5, **kwargs):
    max_faces = int(thr_a)
    min_confidence = thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_mediapipe_face
    if model_mediapipe_face is None:
        from annotator.mediapipe_face import apply_mediapipe_face
        model_mediapipe_face = apply_mediapipe_face
    result = model_mediapipe_face(img, max_faces=max_faces, min_confidence=min_confidence)
    return remove_pad(result), True


model_mlsd = None


def mlsd(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
    thr_v, thr_d = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import apply_mlsd
        model_mlsd = apply_mlsd
    result = model_mlsd(img, thr_v, thr_d)
    return remove_pad(result), True


def unload_mlsd():
    global model_mlsd
    if model_mlsd is not None:
        from annotator.mlsd import unload_mlsd_model
        unload_mlsd_model()


model_depth_anything = None


def depth_anything(img, res:int = 512, colored:bool = True, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_depth_anything
    if model_depth_anything is None:
        with Extra(torch_handler):
            from annotator.depth_anything import DepthAnythingDetector
            device = devices.get_device_for("controlnet")
            model_depth_anything = DepthAnythingDetector(device)
    return remove_pad(model_depth_anything(img, colored=colored)), True


def unload_depth_anything():
    if model_depth_anything is not None:
        model_depth_anything.unload_model()


model_midas = None


def midas(img, res=512, a=np.pi * 2.0, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    result, _ = model_midas(img, a)
    return remove_pad(result), True


def midas_normal(img, res=512, a=np.pi * 2.0, thr_a=0.4, **kwargs):  # bg_th -> thr_a
    bg_th = thr_a
    img, remove_pad = resize_image_with_pad(img, res)
    global model_midas
    if model_midas is None:
        from annotator.midas import apply_midas
        model_midas = apply_midas
    _, result = model_midas(img, a, bg_th)
    return remove_pad(result), True


def unload_midas():
    global model_midas
    if model_midas is not None:
        from annotator.midas import unload_midas_model
        unload_midas_model()


model_leres = None


def leres(img, res=512, a=np.pi * 2.0, thr_a=0, thr_b=0, boost=False, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_leres
    if model_leres is None:
        from annotator.leres import apply_leres
        model_leres = apply_leres
    result = model_leres(img, thr_a, thr_b, boost=boost)
    return remove_pad(result), True


def unload_leres():
    global model_leres
    if model_leres is not None:
        from annotator.leres import unload_leres_model
        unload_leres_model()


class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None

    def run_model(
            self,
            img: np.ndarray,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            use_dw_pose: bool = False,
            use_animal_pose: bool = False,
            json_pose_callback: Callable[[str], None] = None,
            res: int = 512,
            **kwargs  # Ignore rest of kwargs
    ) -> Tuple[np.ndarray, bool]:
        """Run the openpose model. Returns a tuple of
        - result image
        - is_image flag

        The JSON format pose string is passed to `json_pose_callback`.
        """
        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        img, remove_pad = resize_image_with_pad(img, res)

        if self.model_openpose is None:
            from annotator.openpose import OpenposeDetector
            self.model_openpose = OpenposeDetector()

        return remove_pad(self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            use_dw_pose=use_dw_pose,
            use_animal_pose=use_animal_pose,
            json_pose_callback=json_pose_callback
        )), True

    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()


g_openpose_model = OpenposeModel()

model_uniformer = None


def uniformer(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import apply_uniformer
        model_uniformer = apply_uniformer
    result = model_uniformer(img)
    return remove_pad(result), True


def unload_uniformer():
    global model_uniformer
    if model_uniformer is not None:
        from annotator.uniformer import unload_uniformer_model
        unload_uniformer_model()


model_pidinet = None


def pidinet(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img)
    return remove_pad(result), True


def pidinet_ts(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, apply_fliter=True)
    return remove_pad(result), True


def pidinet_safe(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_pidinet
    if model_pidinet is None:
        from annotator.pidinet import apply_pidinet
        model_pidinet = apply_pidinet
    result = model_pidinet(img, is_safe=True)
    return remove_pad(result), True


def scribble_pidinet(img, res=512, **kwargs):
    result, _ = pidinet(img, res)
    import cv2
    from annotator.util import nms
    result = nms(result, 127, 3.0)
    result = cv2.GaussianBlur(result, (0, 0), 3.0)
    result[result > 4] = 255
    result[result < 255] = 0
    return result, True


def unload_pidinet():
    global model_pidinet
    if model_pidinet is not None:
        from annotator.pidinet import unload_pid_model
        unload_pid_model()


clip_encoder = {
    'clip_g': None,
    'clip_h': None,
    'clip_vitl': None,
}


def clip(img, res=512, config='clip_vitl', low_vram=False, **kwargs):
    img = HWC3(img)
    global clip_encoder
    if clip_encoder[config] is None:
        from annotator.clipvision import ClipVisionDetector
        if low_vram:
            logger.info("Loading CLIP model on CPU.")
        clip_encoder[config] = ClipVisionDetector(config, low_vram)
    result = clip_encoder[config](img)
    return result, False


def unload_clip(config='clip_vitl'):
    global clip_encoder
    if clip_encoder[config] is not None:
        clip_encoder[config].unload_model()
        clip_encoder[config] = None


model_color = None


def color(img, res=512, **kwargs):
    img = HWC3(img)
    global model_color
    if model_color is None:
        from annotator.color import apply_color
        model_color = apply_color
    result = model_color(img, res=res)
    return result, True


def lineart_standard(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), 6.0)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    intensity /= max(16, np.median(intensity[intensity > 8]))
    intensity *= 127
    result = intensity.clip(0, 255).astype(np.uint8)
    return remove_pad(result), True


model_lineart = None


def lineart(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector(LineartDetector.model_default)

    # applied auto inversion
    result = 255 - model_lineart(img)
    return remove_pad(result), True


def unload_lineart():
    global model_lineart
    if model_lineart is not None:
        model_lineart.unload_model()


model_lineart_coarse = None


def lineart_coarse(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_coarse
    if model_lineart_coarse is None:
        from annotator.lineart import LineartDetector
        model_lineart_coarse = LineartDetector(LineartDetector.model_coarse)

    # applied auto inversion
    result = 255 - model_lineart_coarse(img)
    return remove_pad(result), True


def unload_lineart_coarse():
    global model_lineart_coarse
    if model_lineart_coarse is not None:
        model_lineart_coarse.unload_model()


model_lineart_anime = None


def lineart_anime(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()

    # applied auto inversion
    result = 255 - model_lineart_anime(img)
    return remove_pad(result), True


def unload_lineart_anime():
    global model_lineart_anime
    if model_lineart_anime is not None:
        model_lineart_anime.unload_model()


model_manga_line = None


def lineart_anime_denoise(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_manga_line
    if model_manga_line is None:
        from annotator.manga_line import MangaLineExtration
        model_manga_line = MangaLineExtration()

    # applied auto inversion
    result = model_manga_line(img)
    return remove_pad(result), True


def unload_lineart_anime_denoise():
    global model_manga_line
    if model_manga_line is not None:
        model_manga_line.unload_model()


model_lama = None


def lama_inpaint(img, res=512, **kwargs):
    H, W, C = img.shape
    raw_color = img[:, :, 0:3].copy()
    raw_mask = img[:, :, 3:4].copy()

    res = 256  # Always use 256 since lama is trained on 256

    img_res, remove_pad = resize_image_with_pad(img, res, skip_hwc3=True)

    global model_lama
    if model_lama is None:
        from annotator.lama import LamaInpainting
        model_lama = LamaInpainting()

    # applied auto inversion
    prd_color = model_lama(img_res)
    prd_color = remove_pad(prd_color)
    prd_color = cv2.resize(prd_color, (W, H))

    alpha = raw_mask.astype(np.float32) / 255.0
    fin_color = prd_color.astype(np.float32) * alpha + raw_color.astype(np.float32) * (1 - alpha)
    fin_color = fin_color.clip(0, 255).astype(np.uint8)

    result = np.concatenate([fin_color, raw_mask], axis=2)

    return result, True


def unload_lama_inpaint():
    global model_lama
    if model_lama is not None:
        model_lama.unload_model()


model_zoe_depth = None


def zoe_depth(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_zoe_depth
    if model_zoe_depth is None:
        from annotator.zoe import ZoeDetector
        model_zoe_depth = ZoeDetector()
    result = model_zoe_depth(img)
    return remove_pad(result), True


def unload_zoe_depth():
    global model_zoe_depth
    if model_zoe_depth is not None:
        model_zoe_depth.unload_model()


model_normal_bae = None


def normal_bae(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_normal_bae
    if model_normal_bae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normal_bae = NormalBaeDetector()
    result = model_normal_bae(img)
    return remove_pad(result), True


def unload_normal_bae():
    global model_normal_bae
    if model_normal_bae is not None:
        model_normal_bae.unload_model()


model_oneformer_coco = None


def oneformer_coco(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_oneformer_coco
    if model_oneformer_coco is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_coco = OneformerDetector(OneformerDetector.configs["coco"])
    result = model_oneformer_coco(img)
    return remove_pad(result), True


def unload_oneformer_coco():
    global model_oneformer_coco
    if model_oneformer_coco is not None:
        model_oneformer_coco.unload_model()


model_oneformer_ade20k = None


def oneformer_ade20k(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_oneformer_ade20k
    if model_oneformer_ade20k is None:
        from annotator.oneformer import OneformerDetector
        model_oneformer_ade20k = OneformerDetector(OneformerDetector.configs["ade20k"])
    result = model_oneformer_ade20k(img)
    return remove_pad(result), True


def unload_oneformer_ade20k():
    global model_oneformer_ade20k
    if model_oneformer_ade20k is not None:
        model_oneformer_ade20k.unload_model()


model_shuffle = None


def shuffle(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    img = remove_pad(img)
    global model_shuffle
    if model_shuffle is None:
        from annotator.shuffle import ContentShuffleDetector
        model_shuffle = ContentShuffleDetector()
    result = model_shuffle(img)
    return result, True


def recolor_luminance(img, res=512, thr_a=1.0, **kwargs):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2LAB)
    result = result[:, :, 0].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result, True


def recolor_intensity(img, res=512, thr_a=1.0, **kwargs):
    result = cv2.cvtColor(HWC3(img), cv2.COLOR_BGR2HSV)
    result = result[:, :, 2].astype(np.float32) / 255.0
    result = result ** thr_a
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result, True


def blur_gaussian(img, res=512, thr_a=1.0, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    img = remove_pad(img)
    result = cv2.GaussianBlur(img, (0, 0), float(thr_a))
    return result, True


model_anime_face_segment = None


def anime_face_segment(img, res=512, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_anime_face_segment
    if model_anime_face_segment is None:
        from annotator.anime_face_segment import AnimeFaceSegment
        model_anime_face_segment = AnimeFaceSegment()

    result = model_anime_face_segment(img)
    return remove_pad(result), True


def unload_anime_face_segment():
    global model_anime_face_segment
    if model_anime_face_segment is not None:
        model_anime_face_segment.unload_model()



def densepose(img, res=512, cmap="viridis", **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    from annotator.densepose import apply_densepose
    result = apply_densepose(img, cmap=cmap)
    return remove_pad(result), True


def unload_densepose():
    from annotator.densepose import unload_model
    unload_model()

model_te_hed = None

def te_hed(img, res=512, thr_a=2, **kwargs):
    img, remove_pad = resize_image_with_pad(img, res)
    global model_te_hed
    if model_te_hed is None:
        from annotator.teed import TEEDDector
        model_te_hed = TEEDDector()
    result = model_te_hed(img, safe_steps=int(thr_a))
    return remove_pad(result), True

def unload_te_hed():
    if model_te_hed is not None:
        model_te_hed.unload_model()

class InsightFaceModel:
    def __init__(self, face_analysis_model_name: str = "buffalo_l"):
        self.model = None
        self.face_analysis_model_name = face_analysis_model_name
        self.antelopev2_installed = False

    def install_antelopev2(self):
        """insightface's github release on antelopev2 model is down. Downloading
        from huggingface mirror."""
        from basicsr.utils.download_util import load_file_from_url
        from annotator.annotator_path import models_path
        model_root = os.path.join(models_path, "insightface", "models", "antelopev2")
        if not model_root:
            os.makedirs(model_root, exist_ok=True)
        for local_file, url in (
            ("1k3d68.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/1k3d68.onnx"),
            ("2d106det.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/2d106det.onnx"),
            ("genderage.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/genderage.onnx"),
            ("glintr100.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/glintr100.onnx"),
            ("scrfd_10g_bnkps.onnx", "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"),
        ):
            local_path = os.path.join(model_root, local_file)
            if not os.path.exists(local_path):
                load_file_from_url(url, model_root)
        self.antelopev2_installed = True

    def load_model(self):
        if self.model is None:
            from insightface.app import FaceAnalysis
            from annotator.annotator_path import models_path
            self.model = FaceAnalysis(
                name=self.face_analysis_model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                root=os.path.join(models_path, "insightface"),
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))

    def run_model(self, img: np.ndarray, **kwargs) -> Tuple[torch.Tensor, bool]:
        self.load_model()
        img = HWC3(img)
        faces = self.model.get(img)
        if not faces:
            raise Exception(f"Insightface: No face found in image {i}.")
        if len(faces) > 1:
            logger.warn("Insightface: More than one face is detected in the image. "
                        f"Only the first one will be used {i}.")
        return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0), False

    def run_model_instant_id(
        self,
        img: np.ndarray,
        res: int = 512,
        return_keypoints: bool = False,
        **kwargs
    ) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
        """Run the insightface model for instant_id.
        Arguments:
            - img: Input image in any size.
            - res: Resolution used to resize image.
            - return_keypoints: Whether to return keypoints image or face embedding.
        """
        def draw_kps(img: np.ndarray, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
            stickwidth = 4
            limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
            kps = np.array(kps)

            h, w, _ = img.shape
            out_img = np.zeros([h, w, 3])

            for i in range(len(limbSeq)):
                index = limbSeq[i]
                color = color_list[index[0]]

                x = kps[index][:, 0]
                y = kps[index][:, 1]
                length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
                polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
            out_img = (out_img * 0.6).astype(np.uint8)

            for idx_kp, kp in enumerate(kps):
                color = color_list[idx_kp]
                x, y = kp
                out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

            return out_img.astype(np.uint8)

        if not self.antelopev2_installed:
            self.install_antelopev2()
        self.load_model()

        img, remove_pad = resize_image_with_pad(img, res)
        face_info = self.model.get(img)
        if not face_info:
            raise Exception(f"Insightface: No face found in image.")
        if len(face_info) > 1:
            logger.warn("Insightface: More than one face is detected in the image. "
                        f"Only the biggest one will be used.")
        # only use the maximum face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
        if return_keypoints:
            return remove_pad(draw_kps(img, face_info['kps'])), True
        else:
            return torch.from_numpy(face_info['embedding']), False


g_insight_face_model = InsightFaceModel()
g_insight_face_instant_id_model = InsightFaceModel(face_analysis_model_name="antelopev2")


@dataclass
class FaceIdPlusInput:
    face_embed: torch.Tensor
    clip_embed: CLIPVisionModelOutput


def face_id_plus(img, low_vram=False, **kwargs):
    """ FaceID plus uses both face_embeding from insightface and clip_embeding from clip. """
    face_embed, _ = g_insight_face_model.run_model(img)
    clip_embed, _ = clip(img, config='clip_h', low_vram=low_vram)
    return FaceIdPlusInput(face_embed, clip_embed), False


class HandRefinerModel:
    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        if self.model is None:
            from annotator.annotator_path import models_path
            from hand_refiner import MeshGraphormerDetector  # installed via hand_refiner_portable
            with Extra(torch_handler):
                self.model = MeshGraphormerDetector.from_pretrained(
                    "hr16/ControlNet-HandRefiner-pruned",
                    cache_dir=os.path.join(models_path, "hand_refiner"),
                    device=self.device,
                )
        else:
            self.model.to(self.device)

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")

    def run_model(self, img, res=512, **kwargs):
        img, remove_pad = resize_image_with_pad(img, res)
        self.load_model()
        with Extra(torch_handler):
            depth_map, mask, info = self.model(
                img, output_type="np",
                detect_resolution=res,
                mask_bbox_padding=30,
            )
        return remove_pad(depth_map), True


g_hand_refiner_model = HandRefinerModel()


model_free_preprocessors = [
    "reference_only",
    "reference_adain",
    "reference_adain+attn",
    "revision_clipvision",
    "revision_ignore_prompt"
]

no_control_mode_preprocessors = [
    "revision_clipvision",
    "revision_ignore_prompt",
    "clip_vision",
    "ip-adapter_clip_sd15",
    "ip-adapter_clip_sdxl",
    "ip-adapter_clip_sdxl_plus_vith",
    "t2ia_style_clipvision",
    "ip-adapter_face_id",
    "ip-adapter_face_id_plus",
]

flag_preprocessor_resolution = "Preprocessor Resolution"
preprocessor_sliders_config = {
    "none": [],
    "inpaint": [],
    "inpaint_only": [],
    "revision_clipvision": [
        None,
        {
            "name": "Noise Augmentation",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0
        },
    ],
    "revision_ignore_prompt": [
        None,
        {
            "name": "Noise Augmentation",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0
        },
    ],
    "canny": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Canny Low Threshold",
            "value": 100,
            "min": 1,
            "max": 255
        },
        {
            "name": "Canny High Threshold",
            "value": 200,
            "min": 1,
            "max": 255
        },
    ],
    "mlsd": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "MLSD Value Threshold",
            "min": 0.01,
            "max": 2.0,
            "value": 0.1,
            "step": 0.01
        },
        {
            "name": "MLSD Distance Threshold",
            "min": 0.01,
            "max": 20.0,
            "value": 0.1,
            "step": 0.01
        }
    ],
    "hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "scribble_hed": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "hed_safe": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "openpose_full": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "dw_openpose_full": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "animal_openpose": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "segmentation": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth_leres": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "depth_leres++": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        },
        {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1,
        }
    ],
    "normal_map": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        },
        {
            "name": "Normal Background Threshold",
            "min": 0.0,
            "max": 1.0,
            "value": 0.4,
            "step": 0.01
        }
    ],
    "threshold": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Binarization Threshold",
            "min": 0,
            "max": 255,
            "value": 127
        }
    ],

    "scribble_xdog": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "XDoG Threshold",
            "min": 1,
            "max": 64,
            "value": 32,
        }
    ],
    "blur_gaussian": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Sigma",
            "min": 0.01,
            "max": 64.0,
            "value": 9.0,
        }
    ],
    "tile_resample": [
        None,
        {
            "name": "Down Sampling Rate",
            "value": 1.0,
            "min": 1.0,
            "max": 8.0,
            "step": 0.01
        }
    ],
    "tile_colorfix": [
        None,
        {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        }
    ],
    "tile_colorfix+sharp": [
        None,
        {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        },
        {
            "name": "Sharpness",
            "value": 1.0,
            "min": 0.0,
            "max": 2.0,
            "step": 0.01
        }
    ],
    "reference_only": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "reference_adain": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "reference_adain+attn": [
        None,
        {
            "name": r'Style Fidelity (only for "Balanced" mode)',
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "inpaint_only+lama": [],
    "color": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        }
    ],
    "mediapipe_face": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048,
        },
        {
            "name": "Max Faces",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        {
            "name": "Min Face Confidence",
            "value": 0.5,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01
        }
    ],
    "recolor_luminance": [
        None,
        {
            "name": "Gamma Correction",
            "value": 1.0,
            "min": 0.1,
            "max": 2.0,
            "step": 0.001
        }
    ],
    "recolor_intensity": [
        None,
        {
            "name": "Gamma Correction",
            "value": 1.0,
            "min": 0.1,
            "max": 2.0,
            "step": 0.001
        }
    ],
    "anime_face_segment": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        }
    ],
    "densepose": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "densepose_parula": [
        {
            "name": flag_preprocessor_resolution,
            "min": 64,
            "max": 2048,
            "value": 512
        }
    ],
    "depth_hand_refiner": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        }
    ],
    "te_hed": [
        {
            "name": flag_preprocessor_resolution,
            "value": 512,
            "min": 64,
            "max": 2048
        },
        {
            "name": "Safe Steps",
            "min": 0,
            "max": 10,
            "value": 2,
            "step": 1,
        },
    ],
}

preprocessor_filters = {
    "All": "none",
    "Canny": "canny",
    "Depth": "depth_midas",
    "NormalMap": "normal_bae",
    "OpenPose": "openpose_full",
    "MLSD": "mlsd",
    "Lineart": "lineart_standard (from white bg & black line)",
    "SoftEdge": "softedge_pidinet",
    "Scribble/Sketch": "scribble_pidinet",
    "Segmentation": "seg_ofade20k",
    "Shuffle": "shuffle",
    "Tile/Blur": "tile_resample",
    "Inpaint": "inpaint_only",
    "InstructP2P": "none",
    "Reference": "reference_only",
    "Recolor": "recolor_luminance",
    "Revision": "revision_clipvision",
    "T2I-Adapter": "none",
    "IP-Adapter": "ip-adapter_clip_sd15",
    "Instant_ID": "instant_id",
}

preprocessor_filters_aliases = {
    'instructp2p': ['ip2p'],
    'segmentation': ['seg'],
    'normalmap': ['normal'],
    't2i-adapter': ['t2i_adapter', 't2iadapter', 't2ia'],
    'ip-adapter': ['ip_adapter', 'ipadapter'],
    'scribble/sketch': ['scribble', 'sketch'],
    'tile/blur': ['tile', 'blur'],
    'openpose':['openpose', 'densepose'],
}  # must use all lower texts
