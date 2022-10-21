from functools import partial

from krita import QCheckBox, QHBoxLayout, QPushButton

from ..script import script
from ..widgets import QComboBoxLayout, QLabel, QSpinBoxLayout
from .img_base import ImgTabBaseWidget


class InpaintTabWidget(ImgTabBaseWidget):
    def __init__(self, *args, **kwargs):
        super(InpaintTabWidget, self).__init__(cfg_prefix="inpaint", *args, **kwargs)
        self.layout.addLayout(self.denoising_strength_layout)

        self.invert_mask = QCheckBox("Invert mask")
        self.mask_blur_layout = QSpinBoxLayout(
            script, "inpaint_mask_blur", "Mask blur (px):", min=0, max=9999, step=1
        )

        inline1 = QHBoxLayout()
        inline1.addWidget(self.invert_mask)
        inline1.addLayout(self.mask_blur_layout)

        self.fill_layout = QComboBoxLayout(
            script, "inpaint_fill_list", "inpaint_fill", label="Inpaint fill:"
        )

        self.full_res = QCheckBox("Inpaint full res")
        self.full_res_padding_layout = QSpinBoxLayout(
            script,
            "inpaint_full_res_padding",
            "Padding (px):",
            min=0,
            max=9999,
            step=1,
        )

        inline2 = QHBoxLayout()
        inline2.addWidget(self.full_res)
        inline2.addLayout(self.full_res_padding_layout)

        self.btn = QPushButton("Start inpaint")

        self.layout.addLayout(inline1)
        self.layout.addLayout(self.fill_layout)
        self.layout.addLayout(inline2)
        self.layout.addWidget(
            QLabel("<em>Tip:</em> Ensure the inpaint layer is selected.")
        )
        self.layout.addWidget(
            QLabel(
                "<em>Tip:</em> Select what the model will see when inpainting. <em>Inpaint full res</em> is unnecessary."
            )
        )
        self.layout.addStretch()
        self.layout.addWidget(self.btn)

    def cfg_init(self):
        super(InpaintTabWidget, self).cfg_init()
        self.mask_blur_layout.cfg_init()
        self.fill_layout.cfg_init()
        self.full_res_padding_layout.cfg_init()
        self.invert_mask.setChecked(script.cfg("inpaint_invert_mask", bool))
        self.full_res.setChecked(script.cfg("inpaint_full_res", bool))

    def cfg_connect(self):
        super(InpaintTabWidget, self).cfg_connect()
        self.mask_blur_layout.cfg_connect()
        self.fill_layout.cfg_connect()
        self.full_res_padding_layout.cfg_connect()

        self.invert_mask.toggled.connect(partial(script.cfg.set, "inpaint_invert_mask"))

        def toggle_fullres(enabled):
            script.cfg.set("inpaint_full_res", enabled)

            # hide/show fullres padding
            self.full_res_padding_layout.qlabel.setVisible(enabled)
            self.full_res_padding_layout.qspin.setVisible(enabled)

        self.full_res.toggled.connect(toggle_fullres)
        toggle_fullres(self.full_res.isChecked())

        self.btn.released.connect(lambda: script.action_inpaint())
