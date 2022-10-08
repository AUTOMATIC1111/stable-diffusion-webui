from functools import partial

from krita import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..defaults import Defaults
from ..script import script


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
            "Override all options with krita_config.yaml (not recommended)"
        )
        self.create_mask_layer = QCheckBox(
            "Create transparency mask layer from selection"
        )
        self.del_temp_files = QCheckBox("Automatically delete temporary image files")
        self.fix_aspect_ratio = QCheckBox("Try to fix aspect ratio for selections")
        self.only_full_img_tiling = QCheckBox(
            "Only allow tiling (on full image) with no selection"
        )

        # webUI/backend settings
        self.filter_nsfw = QCheckBox("Filter NSFW content")
        self.color_correct = QCheckBox(
            "Color correct img2img/inpaint for better blending"
        )
        self.do_exact_steps = QCheckBox(
            "Don't decrease steps based on denoising strength"
        )

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

        layout.addWidget(QLabel("<em>Backend url (Remote URL still broken :p):</em>"))
        layout.addLayout(inline1)

        layout.addWidget(QLabel("<em>Plugin settings:</em>"))
        layout.addWidget(self.just_use_yaml)
        layout.addWidget(self.create_mask_layer)
        layout.addWidget(self.del_temp_files)
        layout.addWidget(self.fix_aspect_ratio)
        layout.addWidget(self.only_full_img_tiling)

        layout.addWidget(QLabel("<em>Backend/webUI settings:</em>"))
        layout.addWidget(self.filter_nsfw)
        layout.addWidget(self.color_correct)
        layout.addWidget(self.do_exact_steps)
        layout.addStretch()
        layout.addWidget(self.restore_defaults)
        layout.addWidget(info_label)

        self.setLayout(layout)

    def cfg_init(self):
        self.base_url.setText(script.cfg("base_url", str))
        self.just_use_yaml.setChecked(script.cfg("just_use_yaml", bool))
        self.create_mask_layer.setChecked(script.cfg("create_mask_layer", bool))
        self.del_temp_files.setChecked(script.cfg("delete_temp_files", bool))
        self.fix_aspect_ratio.setChecked(script.cfg("fix_aspect_ratio", bool))
        self.only_full_img_tiling.setChecked(script.cfg("only_full_img_tiling", bool))
        self.filter_nsfw.setChecked(script.cfg("filter_nsfw", bool))
        self.color_correct.setChecked(script.cfg("color_correct", bool))
        self.do_exact_steps.setChecked(script.cfg("do_exact_steps", bool))

    def cfg_connect(self):
        self.base_url.textChanged.connect(partial(script.set_cfg, "base_url"))
        self.base_url_reset.released.connect(
            lambda: self.base_url.setText(Defaults.base_url)
        )
        self.just_use_yaml.toggled.connect(partial(script.set_cfg, "just_use_yaml"))
        self.create_mask_layer.toggled.connect(
            partial(script.set_cfg, "create_mask_layer")
        )
        self.del_temp_files.toggled.connect(
            partial(script.set_cfg, "delete_temp_files")
        )
        self.fix_aspect_ratio.toggled.connect(
            partial(script.set_cfg, "fix_aspect_ratio")
        )
        self.only_full_img_tiling.toggled.connect(
            partial(script.set_cfg, "only_full_img_tiling")
        )
        self.filter_nsfw.toggled.connect(partial(script.set_cfg, "filter_nsfw"))
        self.color_correct.toggled.connect(partial(script.set_cfg, "color_correct"))
        self.do_exact_steps.toggled.connect(partial(script.set_cfg, "do_exact_steps"))

        def restore_defaults():
            script.restore_defaults()
            # retrieve list of available stuff again
            script.update_config()
            self.update_func()

        self.restore_defaults.released.connect(restore_defaults)
