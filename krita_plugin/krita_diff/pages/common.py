from functools import partial

from krita import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from ..script import script
from ..widgets import QComboBoxLayout, QLabel, QSpinBoxLayout


class SDCommonWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super(SDCommonWidget, self).__init__(*args, **kwargs)

        title = QLabel("<em>Quick Config</em>")

        # Model list
        self.sd_model_layout = QComboBoxLayout(
            script, "sd_model_list", "sd_model", label="SD model:", num_chars=20
        )

        # batch size & count
        self.batch_count_layout = QSpinBoxLayout(
            script, "sd_batch_count", label="Batch count:", min=1, max=9999, step=1
        )
        self.batch_size_layout = QSpinBoxLayout(
            script, "sd_batch_size", label="Batch size:", min=1, max=9999, step=1
        )
        batch_layout = QHBoxLayout()
        batch_layout.addLayout(self.batch_count_layout)
        batch_layout.addLayout(self.batch_size_layout)

        # base/max size adjustment
        self.base_size_layout = QSpinBoxLayout(
            script, "sd_base_size", label="Base size:", min=64, max=8192, step=64
        )
        self.max_size_layout = QSpinBoxLayout(
            script, "sd_max_size", label="Max size:", min=64, max=8192, step=64
        )
        size_layout = QHBoxLayout()
        size_layout.addLayout(self.base_size_layout)
        size_layout.addLayout(self.max_size_layout)

        # global upscaler
        self.upscaler_layout = QComboBoxLayout(
            script, "upscaler_list", "upscaler_name", label="Upscaler:"
        )

        # Restore faces
        self.face_restorer_layout = QComboBoxLayout(
            script,
            "face_restorer_model_list",
            "face_restorer_model",
            label="Face restorer:",
        )
        self.codeformer_weight_layout = QSpinBoxLayout(
            script,
            "codeformer_weight",
            label="CodeFormer weight (max 0, min 1):",
            step=0.01,
        )

        # Tiling mode
        self.tiling = QCheckBox("Tiling mode")

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(self.sd_model_layout)
        layout.addLayout(batch_layout)
        layout.addLayout(size_layout)
        layout.addLayout(self.upscaler_layout)
        layout.addLayout(self.face_restorer_layout)
        layout.addLayout(self.codeformer_weight_layout)
        layout.addWidget(self.tiling)

        self.setLayout(layout)

    def cfg_init(self):
        self.sd_model_layout.cfg_init()
        self.batch_count_layout.cfg_init()
        self.batch_size_layout.cfg_init()
        self.base_size_layout.cfg_init()
        self.max_size_layout.cfg_init()
        self.upscaler_layout.cfg_init()
        self.face_restorer_layout.cfg_init()
        self.codeformer_weight_layout.cfg_init()
        self.tiling.setChecked(script.cfg("sd_tiling", bool))

    def cfg_connect(self):
        self.sd_model_layout.cfg_connect()
        self.batch_count_layout.cfg_connect()
        self.batch_size_layout.cfg_connect()
        self.base_size_layout.cfg_connect()
        self.max_size_layout.cfg_connect()
        self.upscaler_layout.cfg_connect()
        self.face_restorer_layout.cfg_connect()
        self.codeformer_weight_layout.cfg_init()
        self.tiling.toggled.connect(partial(script.cfg.set, "sd_tiling"))

        # Hide codeformer_weight when model isnt codeformer
        def toggle_codeformer_weights(visible):
            self.codeformer_weight_layout.qspin.setVisible(visible)
            self.codeformer_weight_layout.qlabel.setVisible(visible)

        self.face_restorer_layout.qcombo.currentTextChanged.connect(
            lambda t: toggle_codeformer_weights(t == "CodeFormer")
        )
        toggle_codeformer_weights(
            self.face_restorer_layout.qcombo.currentText() == "CodeFormer"
        )
