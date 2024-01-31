import argparse
import unittest
import os
import sys
import time
import datetime
from enum import Enum
from typing import List, Tuple

import cv2
import requests
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


TIMEOUT = 20  # seconds
CWD = os.getcwd()
SKI_IMAGE = os.path.join(CWD, "images/ski.jpg")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_result_dir = os.path.join("results", f"test_result_{timestamp}")
test_expectation_dir = "expectations"
os.makedirs(test_result_dir, exist_ok=True)
os.makedirs(test_expectation_dir, exist_ok=True)
driver_path = ChromeDriverManager().install()


class GenType(Enum):
    txt2img = "txt2img"
    img2img = "img2img"

    def _find_by_xpath(self, driver: webdriver.Chrome, xpath: str) -> "WebElement":
        return driver.find_element(By.XPATH, xpath)

    def tab(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(
            driver,
            f"//*[@id='tabs']/*[contains(@class, 'tab-nav')]//button[text()='{self.value}']",
        )

    def controlnet_panel(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(
            driver, f"//*[@id='tab_{self.value}']//*[@id='controlnet']"
        )

    def generate_button(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(driver, f"//*[@id='{self.value}_generate_box']")

    def prompt_textarea(self, driver: webdriver.Chrome) -> "WebElement":
        return self._find_by_xpath(driver, f"//*[@id='{self.value}_prompt']//textarea")


class SeleniumTestCase(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.driver = None
        self.gen_type = None

    def setUp(self) -> None:
        super().setUp()
        self.driver = webdriver.Chrome(driver_path)
        self.driver.get(webui_url)
        wait = WebDriverWait(self.driver, TIMEOUT)
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#controlnet")))
        self.gen_type = GenType.txt2img

    def tearDown(self) -> None:
        self.driver.quit()
        super().tearDown()

    def select_gen_type(self, gen_type: GenType):
        gen_type.tab(self.driver).click()
        self.gen_type = gen_type

    def set_prompt(self, prompt: str):
        textarea = self.gen_type.prompt_textarea(self.driver)
        textarea.clear()
        textarea.send_keys(prompt)

    def expand_controlnet_panel(self):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        input_image_group = controlnet_panel.find_element(
            By.CSS_SELECTOR, ".cnet-input-image-group"
        )
        if not input_image_group.is_displayed():
            controlnet_panel.click()

    def enable_controlnet_unit(self):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        enable_checkbox = controlnet_panel.find_element(
            By.CSS_SELECTOR, ".cnet-unit-enabled input[type='checkbox']"
        )
        if not enable_checkbox.is_selected():
            enable_checkbox.click()

    def iterate_preprocessor_types(self, ignore_none: bool = True):
        dropdown = self.gen_type.controlnet_panel(self.driver).find_element(
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_controlnet_ControlNet-0_controlnet_preprocessor_dropdown",
        )

        index = 0
        while True:
            dropdown.click()
            options = dropdown.find_elements(
                By.XPATH, "//ul[contains(@class, 'options')]/li"
            )
            input_element = dropdown.find_element(By.CSS_SELECTOR, "input")

            if index >= len(options):
                return

            option = options[index]
            index += 1

            if "none" in option.text and ignore_none:
                continue
            option_text = option.text
            option.click()

            yield option_text

    def select_control_type(self, control_type: str):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        control_type_radio = controlnet_panel.find_element(
            By.CSS_SELECTOR, f'.controlnet_control_type input[value="{control_type}"]'
        )
        control_type_radio.click()
        time.sleep(3)  # Wait for gradio backend to update model/module

    def set_seed(self, seed: int):
        seed_input = self.driver.find_element(
            By.CSS_SELECTOR, f"#{self.gen_type.value}_seed input[type='number']"
        )
        seed_input.clear()
        seed_input.send_keys(seed)

    def set_subseed(self, seed: int):
        show_button = self.driver.find_element(
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_subseed_show input[type='checkbox']",
        )
        if not show_button.is_selected():
            show_button.click()

        subseed_locator = (
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_subseed input[type='number']",
        )
        WebDriverWait(self.driver, TIMEOUT).until(
            EC.visibility_of_element_located(subseed_locator)
        )
        subseed_input = self.driver.find_element(*subseed_locator)
        subseed_input.clear()
        subseed_input.send_keys(seed)

    def upload_controlnet_input(self, img_path: str):
        controlnet_panel = self.gen_type.controlnet_panel(self.driver)
        image_input = controlnet_panel.find_element(
            By.CSS_SELECTOR, '.cnet-input-image-group .cnet-image input[type="file"]'
        )
        image_input.send_keys(img_path)

    def upload_img2img_input(self, img_path: str):
        image_input = self.driver.find_element(
            By.CSS_SELECTOR, '#img2img_image input[type="file"]'
        )
        image_input.send_keys(img_path)

    def generate_image(self, name: str):
        self.gen_type.generate_button(self.driver).click()
        progress_bar_locator_visible = EC.visibility_of_element_located(
            (By.CSS_SELECTOR, f"#{self.gen_type.value}_results .progress")
        )
        WebDriverWait(self.driver, TIMEOUT).until(progress_bar_locator_visible)
        WebDriverWait(self.driver, TIMEOUT * 10).until_not(progress_bar_locator_visible)
        generated_imgs = self.driver.find_elements(
            By.CSS_SELECTOR,
            f"#{self.gen_type.value}_results #{self.gen_type.value}_gallery img",
        )
        for i, generated_img in enumerate(generated_imgs):
            # Use requests to get the image content
            img_content = requests.get(generated_img.get_attribute("src")).content

            # Save the image content to a file
            global overwrite_expectation
            dest_dir = (
                test_expectation_dir if overwrite_expectation else test_result_dir
            )
            img_file_name = f"{self.__class__.__name__}_{name}_{i}.png"
            with open(
                os.path.join(dest_dir, img_file_name),
                "wb",
            ) as img_file:
                img_file.write(img_content)

            if not overwrite_expectation:
                try:
                    img1 = cv2.imread(os.path.join(test_expectation_dir, img_file_name))
                    img2 = cv2.imread(os.path.join(test_result_dir, img_file_name))
                except Exception as e:
                    self.assertTrue(False, f"Get exception reading imgs: {e}")
                    continue

                self.expect_same_image(
                    img1,
                    img2,
                    diff_img_path=os.path.join(
                        test_result_dir, img_file_name.replace(".png", "_diff.png")
                    ),
                )

    def expect_same_image(self, img1, img2, diff_img_path: str):
        # Calculate the difference between the two images
        diff = cv2.absdiff(img1, img2)

        # Set a threshold to highlight the different pixels
        threshold = 30
        diff_highlighted = np.where(diff > threshold, 255, 0).astype(np.uint8)

        # Assert that the two images are similar within a tolerance
        similar = np.allclose(img1, img2, rtol=0.5, atol=1)
        if not similar:
            # Save the diff_highlighted image to inspect the differences
            cv2.imwrite(diff_img_path, diff_highlighted)

        self.assertTrue(similar)


simple_control_types = {
    "Canny": "canny",
    "Depth": "depth_midas",
    "Normal": "normal_bae",
    "OpenPose": "openpose_full",
    "MLSD": "mlsd",
    "Lineart": "lineart_standard (from white bg & black line)",
    "SoftEdge": "softedge_pidinet",
    "Scribble": "scribble_pidinet",
    "Seg": "seg_ofade20k",
    "Tile": "tile_resample",
    # Shuffle and Reference are not stable, and expected to fail.
    # The majority of pixels are same, but some outlier pixels can have big diff.
    "Shuffle": "shuffle",
    "Reference": "reference_only",
}.keys()


class SeleniumTxt2ImgTest(SeleniumTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.select_gen_type(GenType.txt2img)
        self.set_seed(100)
        self.set_subseed(1000)

    def test_simple_control_types(self):
        """Test simple control types that only requires input image."""
        for control_type in simple_control_types:
            with self.subTest(control_type=control_type):
                self.expand_controlnet_panel()
                self.select_control_type(control_type)
                self.upload_controlnet_input(SKI_IMAGE)
                self.generate_image(f"{control_type}_ski")


class SeleniumImg2ImgTest(SeleniumTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.select_gen_type(GenType.img2img)
        self.set_seed(100)
        self.set_subseed(1000)

    def test_simple_control_types(self):
        """Test simple control types that only requires input image."""
        for control_type in simple_control_types:
            with self.subTest(control_type=control_type):
                self.expand_controlnet_panel()
                self.select_control_type(control_type)
                self.upload_img2img_input(SKI_IMAGE)
                self.upload_controlnet_input(SKI_IMAGE)
                self.generate_image(f"img2img_{control_type}_ski")


class SeleniumInpaintTest(SeleniumTestCase):
    def setUp(self) -> None:
        super().setUp()

    def draw_inpaint_mask(self, target_canvas):
        size = target_canvas.size
        width = size["width"]
        height = size["height"]
        brush_radius = 5
        repeat = int(width * 0.1 / brush_radius)

        trace: List[Tuple[int, int]] = [
            (brush_radius, 0),
            (0, height * 0.2),
            (brush_radius, 0),
            (0, -height * 0.2),
        ] * repeat

        actions = ActionChains(self.driver)
        actions.move_to_element(target_canvas)  # move to the canvas
        actions.move_by_offset(*trace[0])
        actions.click_and_hold()  # click and hold the left mouse button down
        for stop_point in trace[1:]:
            actions.move_by_offset(*stop_point)
        actions.release()  # release the left mouse button
        actions.perform()  # perform the action chain

    def draw_cn_mask(self):
        canvas = self.gen_type.controlnet_panel(self.driver).find_element(
            By.CSS_SELECTOR, ".cnet-input-image-group .cnet-image canvas"
        )
        self.draw_inpaint_mask(canvas)

    def draw_a1111_mask(self):
        canvas = self.driver.find_element(By.CSS_SELECTOR, "#img2maskimg canvas")
        self.draw_inpaint_mask(canvas)

    def test_txt2img_inpaint(self):
        self.select_gen_type(GenType.txt2img)
        self.expand_controlnet_panel()
        self.select_control_type("Inpaint")
        self.upload_controlnet_input(SKI_IMAGE)
        self.draw_cn_mask()

        self.set_seed(100)
        self.set_subseed(1000)

        for option in self.iterate_preprocessor_types():
            with self.subTest(option=option):
                self.generate_image(f"{option}_txt2img_ski")

    def test_img2img_inpaint(self):
        # Note: img2img inpaint can only use A1111 mask.
        # ControlNet input is disabled in img2img inpaint.
        self._test_img2img_inpaint(use_cn_mask=False, use_a1111_mask=True)

    def _test_img2img_inpaint(self, use_cn_mask: bool, use_a1111_mask: bool):
        self.select_gen_type(GenType.img2img)
        self.expand_controlnet_panel()
        self.select_control_type("Inpaint")
        self.upload_img2img_input(SKI_IMAGE)
        # Send to inpaint
        self.driver.find_element(
            By.XPATH, f"//*[@id='img2img_copy_to_img2img']//button[text()='inpaint']"
        ).click()
        time.sleep(3)
        # Select latent noise to make inpaint effect more visible.
        self.driver.find_element(
            By.XPATH,
            f"//input[@name='radio-img2img_inpainting_fill' and @value='latent noise']",
        ).click()
        self.set_prompt("(coca-cola:2.0)")
        self.enable_controlnet_unit()
        self.upload_controlnet_input(SKI_IMAGE)

        self.set_seed(100)
        self.set_subseed(1000)

        prefix = ""
        if use_cn_mask:
            self.draw_cn_mask()
            prefix += "controlnet"

        if use_a1111_mask:
            self.draw_a1111_mask()
            prefix += "A1111"

        for option in self.iterate_preprocessor_types():
            with self.subTest(option=option, mask_prefix=prefix):
                self.generate_image(f"{option}_{prefix}_img2img_ski")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument(
        "--overwrite_expectation", action="store_true", help="overwrite expectation"
    )
    parser.add_argument(
        "--target_url", type=str, default="http://localhost:7860", help="WebUI URL"
    )
    args, unknown_args = parser.parse_known_args()
    overwrite_expectation = args.overwrite_expectation
    webui_url = args.target_url

    sys.argv = sys.argv[:1] + unknown_args
    unittest.main()
