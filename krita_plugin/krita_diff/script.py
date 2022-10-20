import json
import math
import os
from dataclasses import asdict
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from krita import Krita, QByteArray, QImage, QObject, QSettings, QTimer

from .defaults import (
    DEFAULTS,
    GET_CONFIG_TIMEOUT,
    POST_TIMEOUT,
    STATE_IMG2IMG,
    STATE_INIT,
    STATE_INPAINT,
    STATE_READY,
    STATE_RESET_DEFAULT,
    STATE_TXT2IMG,
    STATE_UPSCALE,
    STATE_URLERROR,
    STATE_WAIT,
)
from .utils import create_layer, find_optimal_selection_region, fix_prompt


class Script(QObject):
    def __init__(self):
        # Persistent settings (should reload between Krita sessions)
        # NOTE: delete this file between tests, should be in ~/.config/krita/krita_diff_plugin.ini
        self.config = QSettings(
            QSettings.IniFormat, QSettings.UserScope, "krita", "krita_diff_plugin"
        )
        self.restore_defaults(if_empty=True)
        self.is_busy = False

        # Status bar
        self._status_cb = lambda s: None
        self.status = STATE_INIT

    def cfg(self, name: str, type):
        assert self.config.contains(
            name
        ), "Report this bug, developer missed out a config key somewhere."
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

    def update_selection(self):
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

    def adjust_selection(self):
        if self.selection is not None and self.cfg("fix_aspect_ratio", bool):
            x, y, width, height = find_optimal_selection_region(
                self.cfg("sd_base_size", int),
                self.cfg("sd_max_size", int),
                self.x,
                self.y,
                self.width,
                self.height,
                self.doc.width(),
                self.doc.height(),
            )

            self.x = x
            self.y = y
            self.width = width
            self.height = height

    def update_config(self):
        cfg = None
        try:
            with urlopen(
                urljoin(self.cfg("base_url", str), "config"), None, GET_CONFIG_TIMEOUT
            ) as res:
                cfg = json.loads(res.read())
        except Exception as e:
            self.handle_api_error(e)
            return False

        try:
            assert len(cfg["upscalers"]) > 0
            assert len(cfg["samplers"]) > 0
            assert len(cfg["samplers_img2img"]) > 0
            assert len(cfg["face_restorers"]) > 0
            assert len(cfg["sd_models"]) > 0
        except:
            self.set_status(
                f"{STATE_URLERROR}: incompatible response, are you running the right API?"
            )
            return False

        # replace only after verifying
        self.opt = cfg
        self.set_cfg("upscaler_list", self.opt["upscalers"])
        self.set_cfg("txt2img_sampler_list", self.opt["samplers"])
        self.set_cfg("img2img_sampler_list", self.opt["samplers_img2img"])
        self.set_cfg("inpaint_sampler_list", self.opt["samplers_img2img"])
        self.set_cfg("face_restorer_model_list", self.opt["face_restorers"])
        self.set_cfg("sd_model_list", self.opt["sd_models"])
        return True

    def post(self, route, body, base_url=...):
        base_url = self.cfg("base_url", str) if base_url is ... else base_url
        # FastAPI doesn't support urlencoded data transparently
        body = json.dumps(body).encode("utf-8")
        req = Request(urljoin(base_url, route))
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(body)))
        try:
            # TODO: how to cancel this? might as well refactor the API to be async...
            with urlopen(req, body, POST_TIMEOUT) as res:
                return json.loads(res.read())
        except Exception as e:
            self.handle_api_error(e)

    def handle_api_error(self, exc: Exception):
        """Handle exceptions that can occur while interacting with the backend."""
        try:
            # conveniently allows error to bubble back up if not handled by here
            raise exc
        except URLError as e:
            self.set_status(f"{STATE_URLERROR}: {e.reason}")
        except json.JSONDecodeError:
            self.set_status(f"{STATE_URLERROR}: invalid JSON response")
        except ValueError:
            self.set_status(f"{STATE_URLERROR}: Invalid backend URL")

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
            include_grid=self.cfg("include_grid", bool),
        )
        return params

    def post_txt2img(self):
        params = dict(orig_width=self.width, orig_height=self.height)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("txt2img_seed", int)
                if not self.cfg("txt2img_seed", str).strip() == ""
                else -1
            )
            params.update(self.get_common_params())
            params.update(
                prompt=fix_prompt(self.cfg("txt2img_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("txt2img_negative_prompt", str)),
                sampler_name=self.cfg("txt2img_sampler", str),
                steps=self.cfg("txt2img_steps", int),
                cfg_scale=self.cfg("txt2img_cfg_scale", float),
                seed=seed,
                highres_fix=self.cfg("txt2img_highres", bool),
                denoising_strength=self.cfg("txt2img_denoising_strength", float),
            )

        return self.post("/txt2img", params)

    def post_img2img(self, path, mask_path):
        params = dict(mode=0, src_path=path, mask_path=mask_path)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("img2img_seed", int)
                if not self.cfg("img2img_seed", str).strip() == ""
                else -1
            )
            params.update(self.get_common_params())
            params.update(
                prompt=fix_prompt(self.cfg("img2img_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("img2img_negative_prompt", str)),
                sampler_name=self.cfg("img2img_sampler", str),
                steps=self.cfg("img2img_steps", int),
                cfg_scale=self.cfg("img2img_cfg_scale", float),
                denoising_strength=self.cfg("img2img_denoising_strength", float),
                seed=seed,
            )

        return self.post("/img2img", params)

    def post_inpaint(self, path, mask_path):
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
            params.update(self.get_common_params())
            params.update(
                prompt=fix_prompt(self.cfg("inpaint_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("inpaint_negative_prompt", str)),
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
                include_grid=False,  # it is never useful for inpaint mode
            )

        return self.post("/img2img", params)

    def post_upscale(self, path):
        params = (
            {
                "src_path": path,
                "upscaler_name": self.cfg("upscale_upscaler_name", str),
                "downscale_first": self.cfg("upscale_downscale_first", bool),
            }
            if not self.cfg("just_use_yaml", bool)
            else {"src_path": path}
        )
        return self.post("/upscale", params)

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
    def image_to_ba(self, image):
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        return QByteArray(ptr.asstring())

    def insert_img(self, layer_name, path, visible=True):
        image = QImage()
        image.load(path, "PNG")
        ba = self.image_to_ba(image)

        layer = create_layer(layer_name)
        if not visible:
            layer.setVisible(False)

        layer.setPixelData(ba, self.x, self.y, self.width, self.height)
        print(f"Inserted image: {path}")

    def apply_txt2img(self):
        response = self.post_txt2img()
        assert response is not None, "Backend Error, check terminal"
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
            self.post_inpaint(path, mask_path)
            if mode == 1
            else self.post_img2img(path, mask_path)
        )
        assert response is not None, "Backend Error, check terminal"

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

        response = self.post_upscale(path)
        assert response is not None, "Backend Error, check terminal"
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
            self.is_busy = False

    def create_mask_layer_workaround(self):
        if self.cfg("create_mask_layer", bool):
            self.is_busy = True
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
        if self.is_busy:
            pass

        def cb():
            self.update_config()
            self.update_selection()
            self.adjust_selection()
            self.apply_txt2img()
            self.create_mask_layer_workaround()
            self.set_status(STATE_TXT2IMG)

        QTimer.singleShot(1, cb)

    def action_img2img(self):
        self.set_status(STATE_WAIT)
        if self.is_busy:
            pass

        def cb():
            self.update_config()
            self.update_selection()
            self.adjust_selection()
            self.apply_img2img(mode=0)
            self.create_mask_layer_workaround()
            self.set_status(STATE_IMG2IMG)

        QTimer.singleShot(1, cb)

    def action_sd_upscale(self):
        assert False, "disabled"
        self.set_status(STATE_WAIT)
        if self.is_busy:
            pass
        self.update_config()
        self.update_selection()
        self.apply_img2img(mode=2)
        self.create_mask_layer_workaround()

    def action_inpaint(self):
        self.set_status(STATE_WAIT)
        if self.is_busy:
            pass

        def cb():
            self.update_config()
            self.update_selection()
            self.adjust_selection()
            self.apply_img2img(mode=1)
            self.set_status(STATE_INPAINT)

        QTimer.singleShot(1, cb)

    def action_simple_upscale(self):
        self.set_status(STATE_WAIT)
        if self.is_busy:
            pass

        def cb():
            self.update_config()
            self.update_selection()
            self.apply_simple_upscale()
            self.set_status(STATE_UPSCALE)

        QTimer.singleShot(1, cb)


script = Script()
