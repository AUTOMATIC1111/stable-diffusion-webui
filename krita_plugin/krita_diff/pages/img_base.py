from krita import QHBoxLayout, QVBoxLayout, QWidget

from ..script import script
from ..widgets import QComboBoxLayout, QLineEditLayout, QPromptLayout, QSpinBoxLayout


class ImgTabBaseWidget(QWidget):
    def __init__(self, cfg_prefix: str = "", *args, **kwargs):
        super(ImgTabBaseWidget, self).__init__(*args, **kwargs)

        self.prompt_layout = QPromptLayout(
            script, f"{cfg_prefix}_prompt", f"{cfg_prefix}_negative_prompt"
        )

        self.seed_layout = QLineEditLayout(
            script, f"{cfg_prefix}_seed", label="Seed:", placeholder="Random"
        )

        self.sampler_layout = QComboBoxLayout(
            script,
            f"{cfg_prefix}_sampler_list",
            f"{cfg_prefix}_sampler",
            label="Sampler:",
        )

        self.steps_layout = QSpinBoxLayout(
            script, f"{cfg_prefix}_steps", label="Steps:", min=1, max=9999, step=1
        )
        self.cfg_scale_layout = QSpinBoxLayout(
            script, f"{cfg_prefix}_cfg_scale", label="CFG scale:", min=1.0, max=9999.0
        )
        self.script_layout = QComboBoxLayout(
            script,
            f"{cfg_prefix}_script",
            f"{cfg_prefix}_script_list",
            label="(TODO) Scripts:",
        )

        inline_layout = QHBoxLayout()
        inline_layout.addLayout(self.steps_layout)
        inline_layout.addLayout(self.cfg_scale_layout)

        self.layout = layout = QVBoxLayout()
        layout.addLayout(self.script_layout)
        layout.addLayout(self.prompt_layout)
        layout.addLayout(self.seed_layout)
        layout.addLayout(self.sampler_layout)
        layout.addLayout(inline_layout)

        self.setLayout(layout)

        # not added so inheritants can place it wherever they want
        self.denoising_strength_layout = QSpinBoxLayout(
            script,
            f"{cfg_prefix}_denoising_strength",
            label="Denoising strength:",
            step=0.01,
        )

    def cfg_init(self):
        # self.script_layout.cfg_init()
        self.prompt_layout.cfg_init()
        self.seed_layout.cfg_init()
        self.sampler_layout.cfg_init()
        self.steps_layout.cfg_init()
        self.cfg_scale_layout.cfg_init()
        self.denoising_strength_layout.cfg_init()

    def cfg_connect(self):
        # self.script_layout.cfg_connect()
        self.prompt_layout.cfg_connect()
        self.seed_layout.cfg_connect()
        self.sampler_layout.cfg_connect()
        self.steps_layout.cfg_connect()
        self.cfg_scale_layout.cfg_connect()
        self.denoising_strength_layout.cfg_connect()
