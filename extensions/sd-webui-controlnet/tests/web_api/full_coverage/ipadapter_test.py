import unittest
import pytest
from typing import NamedTuple, Optional

from .template import (
    sd_version,
    StableDiffusionVersion,
    is_full_coverage,
    APITestTemplate,
    portrait_imgs,
    realistic_girl_face_img,
    general_negative_prompt,
)


class AdapterSetting(NamedTuple):
    module: str
    model: str
    lora: Optional[str] = None

    @property
    def lora_prompt(self) -> str:
        return f"<lora:{self.lora}:0.6>" if self.lora else ""


# Used to fix pose for better comparison between different settings.
openpose_unit = {
    "module": "openpose",
    "model": (
        "control_v11p_sd15_openpose [cab727d4]"
        if sd_version != StableDiffusionVersion.SDXL
        else "kohya_controllllite_xl_openpose_anime [7e5349e5]"
    ),
    "image": realistic_girl_face_img,
    "weight": 0.8,
}
base_prompt = "1girl, simple background, (white_background: 1.2), portrait"
negative_prompts = {
    "with_neg": general_negative_prompt,
    "no_neg": "",
}


sd15_full_face = AdapterSetting(
    "ip-adapter_clip_sd15",
    "ip-adapter-full-face_sd15 [852b9843]",
)
sd15_plus_face = AdapterSetting(
    "ip-adapter_clip_sd15",
    "ip-adapter-plus-face_sd15 [71693645]",
)
sd15_normal = AdapterSetting(
    "ip-adapter_clip_sd15",
    "ip-adapter_sd15 [6a3f6166]",
)
sd15_light = AdapterSetting(
    "ip-adapter_clip_sd15",
    "ip-adapter_sd15_light [be1c9b97]",
)
sdxl_normal = AdapterSetting(
    "ip-adapter_clip_sdxl",
    "ip-adapter_sdxl [d5d53548]"
)
sdxl_vit = AdapterSetting(
    "ip-adapter_clip_sdxl_plus_vith",
    "ip-adapter_sdxl_vit-h [75a08f84]",
)
sdxl_plus_vit = AdapterSetting(
    "ip-adapter_clip_sdxl_plus_vith",
    "ip-adapter-plus_sdxl_vit-h [f1f19f7d]",
)
sdxl_plus_vit_face = AdapterSetting(
    "ip-adapter_clip_sdxl_plus_vith",
    "ip-adapter-plus-face_sdxl_vit-h [c60d7d48]",
)
class TestIPAdapterFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

        if sd_version == StableDiffusionVersion.SDXL:
            self.settings = [
                sdxl_normal,
                sdxl_vit,
                sdxl_plus_vit,
                sdxl_plus_vit_face,
            ]
        else:
            self.settings = [
                sd15_normal,
                sd15_light,
                sd15_plus_face,
                sd15_full_face,
            ]

    def test_adapter(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name,
                            "txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt},{s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[
                                {
                                    "module": s.module,
                                    "model": s.model,
                                    "image": realistic_girl_face_img,
                                },
                                openpose_unit,
                            ],
                        ).exec()
                    )

    def test_adapter_multi_inputs(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"multi_inputs_{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name=name,
                            gen_type="txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt}, {s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[openpose_unit]
                            + [
                                {
                                    "image": img,
                                    "module": s.module,
                                    "model": s.model,
                                    "weight": 1 / len(portrait_imgs),
                                }
                                for img in portrait_imgs
                            ],
                        ).exec()
                    )

    def test_adapter_real_multi_inputs(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"real_multi_{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name=name,
                            gen_type="txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt}, {s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[
                                openpose_unit,
                                {
                                    "image": [{"image": img} for img in portrait_imgs],
                                    "module": s.module,
                                    "model": s.model,
                                },
                            ],
                        ).exec()
                    )


sd15_face_id = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid_sd15 [0a1757e9]",
    "ip-adapter-faceid_sd15_lora",
)
sd15_face_id_plus = AdapterSetting(
    "ip-adapter_face_id_plus",
    "ip-adapter-faceid-plus_sd15 [d86a490f]",
    "ip-adapter-faceid-plus_sd15_lora",
)
sd15_face_id_plus_v2 = AdapterSetting(
    "ip-adapter_face_id_plus",
    "ip-adapter-faceid-plusv2_sd15 [6e14fc1a]",
    "ip-adapter-faceid-plusv2_sd15_lora",
)
sd15_face_id_portrait = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid-portrait_sd15 [b2609049]",
)
sdxl_face_id = AdapterSetting(
    "ip-adapter_face_id",
    "ip-adapter-faceid_sdxl [59ee31a3]",
    "ip-adapter-faceid_sdxl_lora",
)


class TestIPAdapterFaceIdFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

        if sd_version == StableDiffusionVersion.SDXL:
            self.settings = [sdxl_face_id]
        else:
            self.settings = [
                sd15_face_id,
                sd15_face_id_plus,
                sd15_face_id_plus_v2,
                sd15_face_id_portrait,
            ]

    def test_face_id(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name,
                            "txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt},{s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[
                                {
                                    "module": s.module,
                                    "model": s.model,
                                    "image": realistic_girl_face_img,
                                },
                                openpose_unit,
                            ],
                        ).exec()
                    )

    def test_face_id_multi_inputs(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"multi_inputs_{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name=name,
                            gen_type="txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt}, {s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[openpose_unit]
                            + [
                                {
                                    "image": img,
                                    "module": s.module,
                                    "model": s.model,
                                    "weight": 1 / len(portrait_imgs),
                                }
                                for img in portrait_imgs
                            ],
                        ).exec()
                    )

    def test_face_id_real_multi_inputs(self):
        for s in self.settings:
            for n, negative_prompt in negative_prompts.items():
                name = f"real_multi_{s}_{n}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name=name,
                            gen_type="txt2img",
                            payload_overrides={
                                "prompt": f"{base_prompt}, {s.lora_prompt}",
                                "negative_prompt": negative_prompt,
                                "steps": 20,
                                "width": 512,
                                "height": 512,
                            },
                            unit_overrides=[
                                openpose_unit,
                                {
                                    "image": [{"image": img} for img in portrait_imgs],
                                    "module": s.module,
                                    "model": s.model,
                                },
                            ],
                        ).exec()
                    )


if __name__ == "__main__":
    unittest.main()
