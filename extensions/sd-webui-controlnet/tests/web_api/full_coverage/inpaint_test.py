import unittest
import pytest
from .template import (
    is_full_coverage,
    APITestTemplate,
    girl_img,
    mask_img,
    mask_small_img,
)


class TestInpaintFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()

    def test_inpaint(self):
        for gen_type in ("img2img", "txt2img"):
            if gen_type == "img2img":
                payload = {
                    "init_images": [girl_img],
                    "mask": mask_img,
                }
                unit = {}
            else:
                payload = {}
                unit = {
                    "image": {
                        "image": girl_img,
                        "mask": mask_img,
                    }
                }

            unit["model"] = "control_v11p_sd15_inpaint [ebff9138]"

            for i_resize, resize_mode in enumerate(
                ("Just Resize", "Crop and Resize", "Resize and Fill")
            ):
                # Gen 512x768(input image size) for resize.
                if resize_mode == "Crop and Resize":
                    payload["height"] = 768
                    payload["width"] = 512

                # Gen 512x512 for inner fit.
                if resize_mode == "Crop and Resize":
                    payload["height"] = 512
                    payload["width"] = 512

                # Gen 768x768 for outer fit.
                if resize_mode == "Resize and Fill":
                    payload["height"] = 768
                    payload["width"] = 768

                if gen_type == "img2img":
                    payload["resize_mode"] = i_resize
                else:
                    unit["resize_mode"] = resize_mode

                for module in ("inpaint_only", "inpaint", "inpaint_only+lama"):
                    unit["module"] = module

                    with self.subTest(
                        gen_type=gen_type,
                        resize_mode=resize_mode,
                        module=module,
                    ):
                        self.assertTrue(
                            APITestTemplate(
                                f"{gen_type}_{resize_mode}_{module}",
                                gen_type,
                                payload_overrides=payload,
                                unit_overrides=unit,
                            ).exec()
                        )

    def test_inpaint_no_mask(self):
        """Inpaint should fail if no mask is provided. Output should not contain
        ControlNet detected map."""
        for gen_type in ("img2img", "txt2img"):
            if gen_type == "img2img":
                payload = {
                    "init_images": [girl_img],
                }
                unit = {}
            else:
                payload = {}
                unit = {
                    "image": {
                        "image": girl_img,
                    }
                }
            unit["model"] = "control_v11p_sd15_inpaint [ebff9138]"
            unit["module"] = "inpaint_only"
            with self.subTest(gen_type=gen_type):
                self.assertTrue(
                    APITestTemplate(
                        f"{gen_type}_no_mask_fail",
                        gen_type,
                        payload_overrides=payload,
                        unit_overrides=unit,
                    ).exec()
                )

    def test_inpaint_double_mask(self):
        """When mask is provided for both a1111 img2img input and ControlNet
        unit input, ControlNet input mask should be used."""
        self.assertTrue(
            APITestTemplate(
                f"img2img_double_mask",
                "img2img",
                payload_overrides={
                    "init_images": [girl_img],
                    "mask": mask_img,
                },
                unit_overrides={
                    "image": {
                        "image": girl_img,
                        "mask": mask_small_img,
                    },
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint",
                },
            ).exec()
        )

    def test_img2img_mask_on_unit(self):
        """ Usecase for inpaint_global_harmonious. """
        self.assertTrue(
            APITestTemplate(
                f"img2img_mask_on_unit",
                "img2img",
                payload_overrides={
                    "init_images": [girl_img],
                },
                unit_overrides={
                    "image": {
                        "image": girl_img,
                        "mask": mask_small_img,
                    },
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint",
                },
            ).exec()
        )

    def test_outpaint_without_mask(self):
        self.assertTrue(
            APITestTemplate(
                f"img2img_outpaint_without_mask",
                "img2img",
                payload_overrides={
                    "init_images": [girl_img],
                    "width": 768,
                    "height": 768,
                    "resize_mode": 2,
                },
                unit_overrides={
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint_only+lama",
                },
            ).exec()
        )

        self.assertTrue(
            APITestTemplate(
                f"txt2img_outpaint_without_mask",
                "txt2img",
                payload_overrides={
                    "width": 768,
                    "height": 768,
                },
                unit_overrides={
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint_only+lama",
                    "image": {
                        "image": girl_img,
                    },
                    "resize_mode": 2,
                },
            ).exec()
        )

    def test_inpaint_crop(self):
        self.assertTrue(
            APITestTemplate(
                "img2img_inpaint_crop",
                "img2img",
                payload_overrides={
                    "init_images": [girl_img],
                    "inpaint_full_res": True,
                    "mask": mask_small_img,
                },
                unit_overrides={
                    "model": "control_v11p_sd15_canny [d14c016b]",
                    "module": "canny",
                    "inpaint_crop_input_image": True,
                },
            ).exec()
        )
        self.assertTrue(
            APITestTemplate(
                "img2img_inpaint_no_crop",
                "img2img",
                payload_overrides={
                    "init_images": [girl_img],
                    "inpaint_full_res": True,
                    "mask": mask_small_img,
                },
                unit_overrides={
                    "model": "control_v11p_sd15_canny [d14c016b]",
                    "module": "canny",
                    "inpaint_crop_input_image": False,
                },
            ).exec()
        )


if __name__ == "__main__":
    unittest.main()
