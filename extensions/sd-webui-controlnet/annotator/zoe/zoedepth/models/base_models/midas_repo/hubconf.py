dependencies = ["torch"]

import torch

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small

def DPT_BEiT_L_512(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_L_512 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitl16_512",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_BEiT_L_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_L_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitl16_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_BEiT_B_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_BEiT_B_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="beitb16_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_SwinV2_L_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_SwinV2_L_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="swin2l24_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_SwinV2_B_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_SwinV2_B_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="swin2b24_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_SwinV2_T_256(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_SwinV2_T_256 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="swin2t16_256",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_Swin_L_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_Swin_L_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="swinl12_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_Next_ViT_L_384(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_Next_ViT_L_384 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="next_vit_large_6m",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_LeViT_224(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT_LeViT_224 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def DPT_Large(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT-Large model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="vitl16_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model
    
def DPT_Hybrid(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS DPT-Hybrid model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = DPTDepthModel(
            path=None,
            backbone="vitb_rn50_384",
            non_negative=True,
        )

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model
    
def MiDaS(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS v2.1 model for monocular depth estimation
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet()

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model

def MiDaS_small(pretrained=True, **kwargs):
    """ # This docstring shows up in hub.help()
    MiDaS v2.1 small model for monocular depth estimation on resource-constrained devices
    pretrained (bool): load pretrained weights into model
    """

    model = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

    if pretrained:
        checkpoint = (
            "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
        )
        model.load_state_dict(state_dict)

    return model


def transforms():
    import cv2
    from torchvision.transforms import Compose
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    from midas import transforms

    transforms.default_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.small_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.dpt_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.beit512_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                512,
                512,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.swin384_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.swin256_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                256,
                256,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    transforms.levit_transform = Compose(
        [
            lambda img: {"image": img / 255.0},
            Resize(
                224,
                224,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
            lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        ]
    )

    return transforms
