from krita import QCheckBox, QHBoxLayout, QPushButton

from ..script import script
from ..widgets import QLabel
from .img_base import ImgTabBaseWidget


class Txt2ImgTabWidget(ImgTabBaseWidget):
    def __init__(self, *args, **kwargs):
        super(Txt2ImgTabWidget, self).__init__(cfg_prefix="txt2img", *args, **kwargs)

        self.highres = QCheckBox("Highres fix")

        inline_layout = QHBoxLayout()
        inline_layout.addWidget(self.highres)
        inline_layout.addLayout(self.denoising_strength_layout)

        self.btn = QPushButton("Start txt2img")

        self.layout.addLayout(inline_layout)
        self.layout.addWidget(
            QLabel(
                "<em>Tip:</em> Set base_size and max_size higher for AUTO's txt2img highres fix to work."
            )
        )
        self.layout.addStretch()
        self.layout.addWidget(self.btn)

    def cfg_init(self):
        super(Txt2ImgTabWidget, self).cfg_init()
        self.highres.setChecked(script.cfg("txt2img_highres", bool))

    def cfg_connect(self):
        super(Txt2ImgTabWidget, self).cfg_connect()

        def toggle_highres(enabled):
            script.cfg.set("txt2img_highres", enabled)

            # hide/show denoising strength
            self.denoising_strength_layout.qlabel.setVisible(enabled)
            self.denoising_strength_layout.qspin.setVisible(enabled)

        self.highres.toggled.connect(toggle_highres)
        toggle_highres(self.highres.isChecked())

        self.btn.released.connect(lambda: script.action_txt2img())
