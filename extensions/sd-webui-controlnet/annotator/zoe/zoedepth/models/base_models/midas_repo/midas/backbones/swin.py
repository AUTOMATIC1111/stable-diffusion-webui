import timm

from .swin_common import _make_swin_backbone


def _make_pretrained_swinl12_384(pretrained, hooks=None):
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=pretrained)

    hooks = [1, 1, 17, 1] if hooks == None else hooks
    return _make_swin_backbone(
        model,
        hooks=hooks
    )
