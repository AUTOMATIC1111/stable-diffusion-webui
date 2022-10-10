import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import asdict
from urllib.error import URLError

from krita import Krita, QByteArray, QImage, QObject, QSettings, QTimer

from .defaults import (DEFAULTS, STATE_IMG2IMG, STATE_INIT, STATE_INPAINT,
                       STATE_READY, STATE_RESET_DEFAULT, STATE_TXT2IMG,
                       STATE_UPSCALE, STATE_URLERROR, STATE_WAIT)

# samplers = [
#     "DDIM",
#     "PLMS",
#     "k_dpm_2_a",
#     "k_dpm_2",
#     "k_euler_a",
#     "k_euler",
#     "k_heun",
#     "k_lms",
# ]
# samplers_img2img = [
#     "DDIM",
#     "k_dpm_2_a",
#     "k_dpm_2",
#     "k_euler_a",
#     "k_euler",
#     "k_heun",
#     "k_lms",
# ]
# upscalers = ["None", "Lanczos"]
# face_restorers = ["None", "CodeFormer", "GFPGAN"]


class Script(QObject):
    def __init__(self):
        # Persistent settings (should reload between Krita sessions)
        # NOTE: delete this file between tests, should be in ~/.config/krita/krita_diff_plugin.ini
        self.config = QSettings(
            QSettings.IniFormat, QSettings.UserScope, "krita", "krita_diff_plugin"
        )
        self.restore_defaults(if_empty=True)
        self.working = False

        # Status bar
        self._status_cb = lambda s: None
        self.status = STATE_INIT

    def cfg(self, name: str, type):
        assert self.config.contains(name), "Report this bug, developer missed out a config key somewhere."
        return self.config.value(name, type=type)

    def set_cfg(self, name: str, value, if_empty=False):
        if not if_empty or not self.config.contains(name):
            self.config.setValue(name, value)

    def restore_defaults(self, if_empty=False):
        default = asdict(DEFAULTS)

        for k, v in default.items():
            self.set_cfg(k, v, if_empty)

        if not if_empty:
            self.set_status(STATE_RESET_DEFAULT)

    def set_status_callback(self, cb):
        self._status_cb = cb

    def set_status(self, state):
        self.status = state
        self._status_cb(state)

    def update_config(self):
        res = None
        try:
            with urllib.request.urlopen(self.cfg("base_url", str) + "/config") as req:
                res = req.read()
                self.opt = json.loads(res)
        except URLError:
            self.set_status(STATE_URLERROR)

        # dont update lists if connection failed
        if res:
            assert len(self.opt["upscalers"]) > 0
            assert len(self.opt["samplers"]) > 0
            assert len(self.opt["samplers_img2img"]) > 0
            assert len(self.opt["face_restorers"]) > 0
            assert len(self.opt["sd_models"]) > 0
            self.set_cfg("upscaler_list", self.opt["upscalers"])
            self.set_cfg("txt2img_sampler_list", self.opt["samplers"])
            self.set_cfg("img2img_sampler_list", self.opt["samplers_img2img"])
            self.set_cfg("inpaint_sampler_list", self.opt["samplers_img2img"])
            self.set_cfg("face_restorer_model_list", self.opt["face_restorers"])
            self.set_cfg("sd_model_list", self.opt["sd_models"])

        self.app = Krita.instance()
        self.doc = self.app.activeDocument()
        # self.doc doesnt exist at app startup
        if self.doc:
            self.node = self.doc.activeNode()
            self.selection = self.doc.selection()

            is_not_selected = (
                self.selection is None
                or self.selection.width() < 1
                or self.selection.height() < 1
            )
            if is_not_selected:
                self.x = 0
                self.y = 0
                self.width = self.doc.width()
                self.height = self.doc.height()
            else:
                self.x = self.selection.x()
                self.y = self.selection.y()
                self.width = self.selection.width()
                self.height = self.selection.height()

    # Server API    @staticmethod
    def post(self, url, body):
        req = urllib.request.Request(url)
        req.add_header("Content-Type", "application/json")
        body = json.dumps(body)
        body_encoded = body.encode("utf-8")
        req.add_header("Content-Length", str(len(body_encoded)))
        try:
            with urllib.request.urlopen(req, body_encoded) as res:
                return json.loads(res.read())
        except URLError:
            self.set_status(STATE_URLERROR)

    def get_common_params(self):
        tiling = self.cfg("sd_tiling", bool) and (
            not self.cfg("only_full_img_tiling", bool) or self.selection is None
        )
        # its fine to stuff extra stuff here; pydantic will shave off irrelevant params
        params = dict(
            sd_model=self.cfg("sd_model", str),
            batch_count=self.cfg("sd_batch_count", int),
            batch_size=self.cfg("sd_batch_size", int),
            base_size=self.cfg("sd_base_size", int),
            max_size=self.cfg("sd_max_size", int),
            tiling=tiling,
            upscaler_name=self.cfg("upscaler_name", str),
            restore_faces=self.cfg("face_restorer_model", str) != "None",
            face_restorer=self.cfg("face_restorer_model", str),
            codeformer_weight=self.cfg("codeformer_weight", float),
            filter_nsfw=self.cfg("filter_nsfw", bool),
            color_correct=self.cfg("color_correct", bool),
            do_exact_steps=self.cfg("do_exact_steps", bool),
        )
        return params

    def txt2img(self):
        params = dict(orig_width=self.width, orig_height=self.height)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("txt2img_seed", int)
                if not self.cfg("txt2img_seed", str).strip() == ""
                else -1
            )
            params.update(
                prompt=self.fix_prompt(self.cfg("txt2img_prompt", str)),
                negative_prompt=self.fix_prompt(
                    self.cfg("txt2img_negative_prompt", str)
                ),
                sampler_name=self.cfg("txt2img_sampler", str),
                steps=self.cfg("txt2img_steps", int),
                cfg_scale=self.cfg("txt2img_cfg_scale", float),
                seed=seed,
                highres_fix=self.cfg("txt2img_highres", bool),
                denoising_strength=self.cfg("txt2img_denoising_strength", float),
            )
            params.update(self.get_common_params())

        return self.post(self.cfg("base_url", str) + "/txt2img", params)

    def img2img(self, path, mask_path):
        params = dict(mode=0, src_path=path, mask_path=mask_path)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("img2img_seed", int)
                if not self.cfg("img2img_seed", str).strip() == ""
                else -1
            )
            params.update(
                prompt=self.fix_prompt(self.cfg("img2img_prompt", str)),
                negative_prompt=self.fix_prompt(
                    self.cfg("img2img_negative_prompt", str)
                ),
                sampler_name=self.cfg("img2img_sampler", str),
                steps=self.cfg("img2img_steps", int),
                cfg_scale=self.cfg("img2img_cfg_scale", float),
                denoising_strength=self.cfg("img2img_denoising_strength", float),
                seed=seed,
            )
            params.update(self.get_common_params())

        return self.post(self.cfg("base_url", str) + "/img2img", params)

    def inpaint(self, path, mask_path):
        params = dict(mode=1, src_path=path, mask_path=mask_path)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("inpaint_seed", int)
                if not self.cfg("inpaint_seed", str).strip() == ""
                else -1
            )
            fill = self.cfg("inpaint_fill_list", "QStringList").index(
                self.cfg("inpaint_fill", str)
            )
            params.update(
                prompt=self.fix_prompt(self.cfg("inpaint_prompt", str)),
                negative_prompt=self.fix_prompt(
                    self.cfg("inpaint_negative_prompt", str)
                ),
                sampler_name=self.cfg("inpaint_sampler", str),
                steps=self.cfg("inpaint_steps", int),
                cfg_scale=self.cfg("inpaint_cfg_scale", float),
                denoising_strength=self.cfg("inpaint_denoising_strength", float),
                seed=seed,
                invert_mask=self.cfg("inpaint_invert_mask", bool),
                mask_blur=self.cfg("inpaint_mask_blur", int),
                inpainting_fill=fill,
                inpaint_full_res=self.cfg("inpaint_full_res", bool),
                inpaint_full_res_padding=self.cfg("inpaint_full_res_padding", int),
            )
            params.update(self.get_common_params())

        return self.post(self.cfg("base_url", str) + "/img2img", params)

    def simple_upscale(self, path):
        params = (
            {
                "src_path": path,
                "upscaler_name": self.cfg("upscale_upscaler_name", str),
                "downscale_first": self.cfg("upscale_downscale_first", bool),
            }
            if not self.cfg("just_use_yaml", bool)
            else {"src_path": path}
        )
        return self.post(self.cfg("base_url", str) + "/upscale", params)

    def fix_prompt(self, prompt):
        joined = ", ".join(filter(bool, [x.strip() for x in prompt.splitlines()]))
        return joined if joined != "" else None

    def find_final_aspect_ratio(self):
        base_size = self.cfg("sd_base_size", int)
        max_size = self.cfg("sd_max_size", int)

        def rnd(r, x):
            z = 64
            # TODO: for selections with extreme ratios, it might round to 0, causing zero devision
            # however, this temporary fix will return the wrong aspect ratio instead of actually
            # fixing the problem (i.e. warning the user or resetting the box)
            return z * max(round(r * x / z), 1)

        ratio = self.width / self.height

        if self.width > self.height:
            width, height = rnd(ratio, base_size), base_size
            if width > max_size:
                width, height = max_size, rnd(1 / ratio, max_size)
        else:
            width, height = base_size, rnd(1 / ratio, base_size)
            if height > max_size:
                width, height = rnd(ratio, max_size), max_size

        return width / height

    def try_fix_aspect_ratio(self):
        # SD will need to resize image to match required size of 512x(512 + 64*j). That may fuck aspect ratio.
        # That's why we will try to make selection slightly bigger to unfuck aspect ratio a little

        if self.selection is not None and self.cfg("fix_aspect_ratio", bool):
            ratio = self.width / self.height
            final_ratio = self.find_final_aspect_ratio()

            delta = abs(final_ratio - ratio)
            delta_rev = abs(1 / final_ratio - 1 / ratio)
            x_limit = math.ceil(delta * self.width)
            y_limit = math.ceil(delta_rev * self.height)

            best_delta = delta
            best_x1, best_y1 = self.x, self.y
            best_x2, best_y2 = self.x + self.width, self.y + self.height
            for x in range(x_limit):
                x1 = max(0, self.x - (x // 2))
                x2 = min(self.doc.width(), x1 + self.width + x)
                for y in range(y_limit):
                    y1 = max(0, self.y - (y // 2))
                    y2 = min(self.doc.height(), y1 + self.height + y)

                    curr_ratio = (x2 - x1) / (y2 - y1)
                    curr_delta = abs(curr_ratio - final_ratio)
                    if curr_delta < best_delta:
                        best_delta = curr_delta
                        best_x1, best_y1 = x1, y1
                        best_x2, best_y2 = x2, y2

            self.x = best_x1
            self.y = best_y1
            self.width = best_x2 - best_x1
            self.height = best_y2 - best_y1

    def save_img(self, path, is_mask=False):
        if is_mask:
            pixel_bytes = self.node.pixelData(self.x, self.y, self.width, self.height)
        else:
            pixel_bytes = self.doc.pixelData(self.x, self.y, self.width, self.height)

        image_data = QImage(
            pixel_bytes, self.width, self.height, QImage.Format_RGBA8888
        ).rgbSwapped()
        image_data.save(path, "PNG", self.cfg("png_quality", int))
        print(f"Saved {'mask' if is_mask else 'image'}: {path}")

    # Krita tools
    def create_layer(self, name):
        root = self.doc.rootNode()
        layer = self.doc.createNode(name, "paintLayer")
        root.addChildNode(layer, None)
        print(f"created layer: {layer}")
        return layer

    def image_to_ba(self, image):
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        return QByteArray(ptr.asstring())

    def insert_img(self, layer_name, path, visible=True):
        image = QImage()
        image.load(path, "PNG")
        ba = self.image_to_ba(image)

        layer = self.create_layer(layer_name)
        if not visible:
            layer.setVisible(False)

        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"Inserted image: {path}")

    def apply_txt2img(self):
        response = self.txt2img()
        outputs = response["outputs"]
        print(f"Getting images: {outputs}")
        for i, output in enumerate(outputs):
            self.insert_img(
                f"txt2img {i + 1}: {os.path.basename(output)}",
                output,
                i + 1 == len(outputs),
            )
        self.clear_temp_images(outputs)
        self.doc.refreshProjection()

    def apply_img2img(self, mode):
        path = self.opt["new_img"]
        mask_path = self.opt["new_img_mask"]
        self.save_img(path)
        if mode == 1:
            self.save_img(mask_path, is_mask=True)

        response = (
            self.inpaint(path, mask_path)
            if mode == 1
            else self.img2img(path, mask_path)
        )

        outputs = response["outputs"]
        print(f"Getting images: {outputs}")
        layer_name_prefix = (
            "inpaint" if mode == 1 else "sd upscale" if mode == 2 else "img2img"
        )
        for i, output in enumerate(outputs):
            self.insert_img(
                f"{layer_name_prefix} {i + 1}: {os.path.basename(output)}",
                output,
                i + 1 == len(outputs),
            )

        if mode == 1:
            self.clear_temp_images([path, mask_path, *outputs])
        else:
            self.clear_temp_images([path, *outputs])

        self.doc.refreshProjection()

    def apply_simple_upscale(self):
        path = self.opt["new_img"]
        self.save_img(path)

        response = self.simple_upscale(path)
        output = response["output"]
        print(f"Getting image: {output}")

        self.insert_img(f"upscale: {os.path.basename(output)}", output)
        self.clear_temp_images([path, output])
        self.doc.refreshProjection()

    def create_mask_layer_internal(self):
        try:
            if self.selection is not None:
                self.app.action("add_new_transparency_mask").trigger()
                print(f"created mask layer")
                self.doc.setSelection(self.selection)
        finally:
            self.working = False

    def create_mask_layer_workaround(self):
        if self.cfg("create_mask_layer", bool):
            self.working = True
            QTimer.singleShot(
                self.cfg("workaround_timeout", int),
                lambda: self.create_mask_layer_internal(),
            )

    def clear_temp_images(self, files):
        if self.cfg("delete_temp_files", bool):
            for file in files:
                os.remove(file)

    # Actions
    def action_txt2img(self):
        self.set_status(STATE_WAIT)
        if self.working:
            pass

        def cb():
            self.update_config()
            self.try_fix_aspect_ratio()
            self.apply_txt2img()
            self.create_mask_layer_workaround()
            self.set_status(STATE_TXT2IMG)

        QTimer.singleShot(1, cb)

    def action_img2img(self):
        self.set_status(STATE_WAIT)
        if self.working:
            pass

        def cb():
            self.update_config()
            self.try_fix_aspect_ratio()
            self.apply_img2img(mode=0)
            self.create_mask_layer_workaround()
            self.set_status(STATE_IMG2IMG)

        QTimer.singleShot(1, cb)

    def action_sd_upscale(self):
        assert False, "disabled"
        self.set_status(STATE_WAIT)
        if self.working:
            pass
        self.update_config()
        self.apply_img2img(mode=2)
        self.create_mask_layer_workaround()

    def action_inpaint(self):
        self.set_status(STATE_WAIT)
        if self.working:
            pass

        def cb():
            self.update_config()
            self.try_fix_aspect_ratio()
            self.apply_img2img(mode=1)
            self.set_status(STATE_INPAINT)

        QTimer.singleShot(1, cb)

    def action_simple_upscale(self):
        self.set_status(STATE_WAIT)
        if self.working:
            pass

        def cb():
            self.update_config()
            self.apply_simple_upscale()
            self.set_status(STATE_UPSCALE)

        QTimer.singleShot(1, cb)


script = Script()
