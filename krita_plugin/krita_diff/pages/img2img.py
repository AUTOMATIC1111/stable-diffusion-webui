from krita import QPushButton

from ..script import script
from .img_base import ImgTabBaseWidget


# TODO: Add necessary UI components to match up with upstream changes.
class Img2ImgTabWidget(ImgTabBaseWidget):
    def __init__(self, *args, **kwargs):
        super(Img2ImgTabWidget, self).__init__(cfg_prefix="img2img", *args, **kwargs)

        self.btn = QPushButton("Start img2img")

        self.layout.addLayout(self.denoising_strength_layout)
        self.layout.addStretch()
        self.layout.addWidget(self.btn)

    def cfg_init(self):
        super(Img2ImgTabWidget, self).cfg_init()

    def cfg_connect(self):
        super(Img2ImgTabWidget, self).cfg_connect()
        self.btn.released.connect(lambda: script.action_img2img())
