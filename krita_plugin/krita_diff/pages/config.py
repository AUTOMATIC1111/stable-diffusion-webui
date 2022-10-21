from functools import partial

from krita import QCheckBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..defaults import DEFAULTS, STATE_READY
from ..script import script
from ..widgets import QLabel


class ConfigTabWidget(QWidget):
    def __init__(self, update_func, *args, **kwargs):
        super(ConfigTabWidget, self).__init__(*args, **kwargs)

        # callback to update all the other widgets
        self.update_func = update_func

        self.base_url = QLineEdit()
        self.base_url_reset = QPushButton("Default")
        inline1 = QHBoxLayout()
        inline1.addWidget(self.base_url)
        inline1.addWidget(self.base_url_reset)

        # Plugin settings
        self.just_use_yaml = QCheckBox(
            "Override with krita_config.yaml (unrecommended)"
        )
        self.create_mask_layer = QCheckBox("Create transparency mask from selection")
        self.del_temp_files = QCheckBox("Auto delete debug image files")
        self.fix_aspect_ratio = QCheckBox("Fix aspect ratio for selections")
        self.only_full_img_tiling = QCheckBox("Only allow tiling with no selection")
        self.include_grid = QCheckBox("Include grid for txt2img and img2img")

        # webUI/backend settings
        self.filter_nsfw = QCheckBox("Filter NSFW content")
        self.img2img_color_correct = QCheckBox(
            "Color correct img2img for better blending"
        )
        self.inpaint_color_correct = QCheckBox(
            "Color correct inpaint for better blending"
        )
        self.do_exact_steps = QCheckBox(
            "Don't decrease steps based on denoising strength"
        )

        self.refresh_btn = QPushButton("Auto-Refresh Options Now")
        self.restore_defaults = QPushButton("Restore Defaults")

        info_label = QLabel(
            """
            <em>Tip:</em> Only a selected few backend/webUI settings are exposed above.<br/>
            <em>Tip:</em> You should look through & configure all the backend/webUI settings at least once.
            <br/><br/>
            <a href="http://127.0.0.1:7860/" target="_blank">Configure all settings in webUI</a><br/>
            <a href="https://github.com/Interpause/auto-sd-krita/wiki" target="_blank">Read the guide</a><br/>
            <a href="https://github.com/Interpause/auto-sd-krita/issues" target="_blank">Report bugs or suggest features</a>
            """
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("<em>Backend url:</em>"))
        layout.addLayout(inline1)

        layout.addWidget(QLabel("<em>Plugin settings:</em>"))
        layout.addWidget(self.just_use_yaml)
        layout.addWidget(self.create_mask_layer)
        layout.addWidget(self.del_temp_files)
        layout.addWidget(self.fix_aspect_ratio)
        layout.addWidget(self.only_full_img_tiling)
        layout.addWidget(self.include_grid)

        layout.addWidget(QLabel("<em>Backend/webUI settings:</em>"))
        layout.addWidget(self.filter_nsfw)
        layout.addWidget(self.img2img_color_correct)
        layout.addWidget(self.inpaint_color_correct)
        layout.addWidget(self.do_exact_steps)
        layout.addStretch()
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.restore_defaults)
        layout.addWidget(info_label)

        self.setLayout(layout)

    def cfg_init(self):
        # NOTE: update timer -> cfg_init, setText seems to reset cursor position so we prevent it
        base_url = script.cfg("base_url", str)
        if self.base_url.text() != base_url:
            self.base_url.setText(base_url)

        self.just_use_yaml.setChecked(script.cfg("just_use_yaml", bool))
        self.create_mask_layer.setChecked(script.cfg("create_mask_layer", bool))
        self.del_temp_files.setChecked(script.cfg("delete_temp_files", bool))
        self.fix_aspect_ratio.setChecked(script.cfg("fix_aspect_ratio", bool))
        self.only_full_img_tiling.setChecked(script.cfg("only_full_img_tiling", bool))
        self.include_grid.setChecked(script.cfg("include_grid", bool))
        self.filter_nsfw.setChecked(script.cfg("filter_nsfw", bool))
        self.img2img_color_correct.setChecked(script.cfg("img2img_color_correct", bool))
        self.inpaint_color_correct.setChecked(script.cfg("inpaint_color_correct", bool))
        self.do_exact_steps.setChecked(script.cfg("do_exact_steps", bool))

    def cfg_connect(self):
        self.base_url.textChanged.connect(partial(script.cfg.set, "base_url"))
        self.base_url_reset.released.connect(
            lambda: self.base_url.setText(DEFAULTS.base_url)
        )
        self.just_use_yaml.toggled.connect(partial(script.cfg.set, "just_use_yaml"))
        self.create_mask_layer.toggled.connect(
            partial(script.cfg.set, "create_mask_layer")
        )
        self.del_temp_files.toggled.connect(
            partial(script.cfg.set, "delete_temp_files")
        )
        self.fix_aspect_ratio.toggled.connect(
            partial(script.cfg.set, "fix_aspect_ratio")
        )
        self.only_full_img_tiling.toggled.connect(
            partial(script.cfg.set, "only_full_img_tiling")
        )
        self.include_grid.toggled.connect(partial(script.cfg.set, "include_grid"))
        self.filter_nsfw.toggled.connect(partial(script.cfg.set, "filter_nsfw"))
        self.img2img_color_correct.toggled.connect(
            partial(script.cfg.set, "img2img_color_correct")
        )
        self.inpaint_color_correct.toggled.connect(
            partial(script.cfg.set, "inpaint_color_correct")
        )
        self.do_exact_steps.toggled.connect(partial(script.cfg.set, "do_exact_steps"))

        def update_remote_config():
            if script.update_config():
                script.set_status(STATE_READY)

            self.update_func()

        def restore_defaults():
            script.restore_defaults()
            # retrieve list of available stuff again
            script.update_config()
            self.update_func()

        self.refresh_btn.released.connect(update_remote_config)
        self.restore_defaults.released.connect(restore_defaults)
