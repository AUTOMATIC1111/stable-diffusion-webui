import importlib
utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts.global_state import select_control_type, ui_preprocessor_keys
from scripts.enums import StableDiffusionVersion


dummy_value = "dummy"
cn_models = {
    "None": dummy_value,
    "canny_sd15": dummy_value,
    "canny_sdxl": dummy_value,
}


# Tests for the select_control_type function
class TestSelectControlType:
    def test_all_control_type(self):
        result = select_control_type("All", cn_models=cn_models)
        assert result == (
            [ui_preprocessor_keys, list(cn_models.keys()), "none", "None"]
        ), "Expected all preprocessors and models"

    def test_sd_version(self):
        (_, filtered_model_list, _, default_model) = select_control_type(
            "Canny", sd_version=StableDiffusionVersion.UNKNOWN, cn_models=cn_models
        )
        assert filtered_model_list == [
            "None",
            "canny_sd15",
            "canny_sdxl",
        ], "UNKNOWN sd version should match all models"
        assert default_model == "canny_sd15"

        (_, filtered_model_list, _, default_model) = select_control_type(
            "Canny", sd_version=StableDiffusionVersion.SD1x, cn_models=cn_models
        )
        assert filtered_model_list == [
            "None",
            "canny_sd15",
        ], "sd1x version should only sd1x"
        assert default_model == "canny_sd15"

        (_, filtered_model_list, _, default_model) = select_control_type(
            "Canny", sd_version=StableDiffusionVersion.SDXL, cn_models=cn_models
        )
        assert filtered_model_list == [
            "None",
            "canny_sdxl",
        ], "sdxl version should only sdxl"
        assert default_model == "canny_sdxl"

    def test_invert_preprocessor(self):
        for control_type in ("Canny", "Lineart", "Scribble/Sketch", "MLSD"):
            filtered_preprocessor_list, _, _, _ = select_control_type(
                control_type, cn_models=cn_models
            )
            assert any(
                "invert" in module.lower() for module in filtered_preprocessor_list
            )

    def test_no_module_available(self):
        (_, filtered_model_list, _, default_model) = select_control_type(
            "Depth", cn_models=cn_models
        )
        assert filtered_model_list == ["None"]
        assert default_model == "None"
