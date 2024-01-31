from enum import Enum
from typing import Any


class StableDiffusionVersion(Enum):
    """The version family of stable diffusion model."""

    UNKNOWN = 0
    SD1x = 1
    SD2x = 2
    SDXL = 3

    @staticmethod
    def detect_from_model_name(model_name: str) -> "StableDiffusionVersion":
        """Based on the model name provided, guess what stable diffusion version it is.
        This might not be accurate without actually inspect the file content.
        """
        if any(f"sd{v}" in model_name.lower() for v in ("14", "15", "16")):
            return StableDiffusionVersion.SD1x

        if "sd21" in model_name or "2.1" in model_name:
            return StableDiffusionVersion.SD2x

        if "xl" in model_name.lower():
            return StableDiffusionVersion.SDXL

        return StableDiffusionVersion.UNKNOWN

    def encoder_block_num(self) -> int:
        if self in (StableDiffusionVersion.SD1x, StableDiffusionVersion.SD2x, StableDiffusionVersion.UNKNOWN):
            return 12
        else:
            return 9 # SDXL

    def controlnet_layer_num(self) -> int:
        return self.encoder_block_num() + 1

    def is_compatible_with(self, other: "StableDiffusionVersion") -> bool:
        """ Incompatible only when one of version is SDXL and other is not. """
        return (
            any(v == StableDiffusionVersion.UNKNOWN for v in [self, other]) or
            sum(v == StableDiffusionVersion.SDXL for v in [self, other]) != 1
        )


class ControlModelType(Enum):
    """
    The type of Control Models (supported or not).
    """

    ControlNet = "ControlNet, Lvmin Zhang"
    T2I_Adapter = "T2I_Adapter, Chong Mou"
    T2I_StyleAdapter = "T2I_StyleAdapter, Chong Mou"
    T2I_CoAdapter = "T2I_CoAdapter, Chong Mou"
    MasaCtrl = "MasaCtrl, Mingdeng Cao"
    GLIGEN = "GLIGEN, Yuheng Li"
    AttentionInjection = "AttentionInjection, Lvmin Zhang"  # A simple attention injection written by Lvmin
    StableSR = "StableSR, Jianyi Wang"
    PromptDiffusion = "PromptDiffusion, Zhendong Wang"
    ControlLoRA = "ControlLoRA, Wu Hecong"
    ReVision = "ReVision, Stability"
    IPAdapter = "IPAdapter, Hu Ye"
    Controlllite = "Controlllite, Kohya"
    InstantID = "InstantID, Qixun Wang"

    def is_controlnet(self) -> bool:
        """Returns whether the control model should be treated as ControlNet."""
        return self in (
            ControlModelType.ControlNet,
            ControlModelType.ControlLoRA,
            ControlModelType.InstantID,
        )

    def allow_context_sharing(self) -> bool:
        """Returns whether this control model type allows the same PlugableControlModel
        object map to multiple ControlNetUnit.
        Both IPAdapter and Controlllite have unit specific input (clip/image) stored
        on the model object during inference. Sharing the context means that the input
        set earlier gets lost.
        """
        return self not in (
            ControlModelType.IPAdapter,
            ControlModelType.Controlllite,
        )

# Written by Lvmin
class AutoMachine(Enum):
    """
    Lvmin's algorithm for Attention/AdaIn AutoMachine States.
    """

    Read = "Read"
    Write = "Write"
    StyleAlign = "StyleAlign"


class HiResFixOption(Enum):
    BOTH = "Both"
    LOW_RES_ONLY = "Low res only"
    HIGH_RES_ONLY = "High res only"

    @staticmethod
    def from_value(value: Any) -> "HiResFixOption":
        if isinstance(value, str) and value.startswith("HiResFixOption."):
            _, field = value.split(".")
            return getattr(HiResFixOption, field)
        if isinstance(value, str):
            return HiResFixOption(value)
        elif isinstance(value, int):
            return [x for x in HiResFixOption][value]
        else:
            assert isinstance(value, HiResFixOption)
            return value


class InputMode(Enum):
    # Single image to a single ControlNet unit.
    SIMPLE = "simple"
    # Input is a directory. N generations. Each generation takes 1 input image
    # from the directory.
    BATCH = "batch"
    # Input is a directory. 1 generation. Each generation takes N input image
    # from the directory.
    MERGE = "merge"
