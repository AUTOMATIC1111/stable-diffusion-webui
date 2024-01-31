import cv2
import torch

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

default_models = {
    "dpt_beit_large_512": "weights/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "weights/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "weights/dpt_beit_base_384.pt",
    "dpt_swin2_large_384": "weights/dpt_swin2_large_384.pt",
    "dpt_swin2_base_384": "weights/dpt_swin2_base_384.pt",
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",
    "dpt_swin_large_384": "weights/dpt_swin_large_384.pt",
    "dpt_next_vit_large_384": "weights/dpt_next_vit_large_384.pt",
    "dpt_levit_224": "weights/dpt_levit_224.pt",
    "dpt_large_384": "weights/dpt_large_384.pt",
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",
    "midas_v21_384": "weights/midas_v21_384.pt",
    "midas_v21_small_256": "weights/midas_v21_small_256.pt",
    "openvino_midas_v21_small_256": "weights/openvino_midas_v21_small_256.xml",
}


def load_model(device, model_path, model_type="dpt_large_384", optimize=True, height=None, square=False):
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?

    Returns:
        The loaded network, the transform which prepares images as input to the network and the dimensions of the
        network input
    """
    if "openvino" in model_type:
        from openvino.runtime import Core

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_512",
            non_negative=True,
        )
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_beit_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="beitb16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_base_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2b24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_swin_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swinl12_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_next_vit_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="next_vit_large_6m",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # We change the notation from dpt_levit_224 (MiDaS notation) to levit_384 (timm notation) here, where the 224 refers
    # to the resolution 224x224 used by LeViT and 384 is the first entry of the embed_dim, see _cfg and model_cfgs of
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py
    # (commit id: 927f031293a30afb940fff0bee34b85d9c059b0e)
    elif model_type == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small_256":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "openvino_midas_v21_small_256":
        ie = Core()
        uncompiled_model = ie.read_model(model=model_path)
        model = ie.compile_model(uncompiled_model, "CPU")
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    if not "openvino" in model_type:
        print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    else:
        print("Model loaded, optimized with OpenVINO")

    if "openvino" in model_type:
        keep_aspect_ratio = False

    if height is not None:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if not "openvino" in model_type:
        model.eval()

    if optimize and (device == torch.device("cuda")):
        if not "openvino" in model_type:
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
        else:
            print("Error: OpenVINO models are already optimized. No optimization to half-float possible.")
            exit()

    if not "openvino" in model_type:
        model.to(device)

    return model, transform, net_w, net_h
