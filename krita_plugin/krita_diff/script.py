import os

from krita import Krita, QByteArray, QImage, QObject, QTimer

from .client import Client
from .config import Config
from .defaults import (
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
from .utils import create_layer, find_optimal_selection_region


# Does it actually have to be a QObject?
# The only possible use I see is for event emitting
class Script(QObject):
    def __init__(self):
        # Persistent settings (should reload between Krita sessions)
        # NOTE: delete this file between tests, should be in ~/.config/krita/krita_diff_plugin.ini
        self.config = Config()
        self.client = Client(self.config, lambda s: self.set_status(s))
        self.is_busy = False

        # Status bar
        self._status_cb = lambda s: None
        self.status = STATE_INIT

    def cfg(self, name: str, type):
        return self.config.get(name, type)

    def set_cfg(self, name: str, value, if_empty=False):
        return self.config.set(name, value, not if_empty)

    def restore_defaults(self, if_empty=False):
        self.config.restore_defaults(not if_empty)

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
        return self.client.get_config()

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
        response = self.client.post_txt2img(
            self.width, self.height, self.selection is not None
        )
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
        path = self.cfg("new_img_path", str)
        mask_path = self.cfg("new_img_mask_path", str)
        self.save_img(path)
        if mode == 1:
            self.save_img(mask_path, is_mask=True)

        response = (
            self.client.post_inpaint(path, mask_path, self.selection is not None)
            if mode == 1
            else self.client.post_img2img(path, mask_path, self.selection is not None)
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
        path = self.cfg("new_img_path", str)
        self.save_img(path)

        response = self.client.post_upscale(path)
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
