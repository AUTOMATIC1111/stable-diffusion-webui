from functools import partial
from urllib.error import URLError

from .script import *
from .widgets import QComboBoxLayout, QPromptLayout


class SDPluginDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SD Plugin")
        self.create_interface()

        script.update_config()

        self.init_txt2img_interface()
        self.init_img2img_interface()
        self.init_upscale_interface()
        self.init_config_interface()

        self.connect_txt2img_interface()
        self.connect_img2img_interface()
        self.connect_upscale_interface()
        self.connect_config_interface()

        self.setWidget(self.widget)

    def create_interface(self):
        self.create_txt2img_interface()
        self.create_img2img_interface()
        self.create_upscale_interface()
        self.create_config_interface()

        refresh = QPushButton("Refresh Available Options")
        refresh.released.connect(lambda: self.update_remote_config())

        self.tabs = QTabWidget()
        self.tabs.addTab(self.img2img_widget, "Img2Img")
        self.tabs.addTab(self.upscale_widget, "Upscale")
        self.tabs.addTab(self.txt2img_widget, "Txt2Img")
        self.tabs.addTab(self.config_widget, "Config")

        self.layout = QVBoxLayout()
        self.layout.addWidget(refresh)
        self.layout.addWidget(self.tabs)
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout)

    # TODO: Add necessary UI components to match up with upstream changes.
    def create_txt2img_interface(self):
        self.txt2img_prompt = QPromptLayout(
            script, "txt2img_prompt", "txt2img_negative_prompt"
        )
        self.txt2img_sampler = QComboBoxLayout(
            script, "txt2img_sampler_list", "txt2img_sampler", label="Sampler:"
        )

        self.txt2img_steps_label = QLabel("Steps:")
        self.txt2img_steps = QSpinBox()
        self.txt2img_steps.setMinimum(1)
        self.txt2img_steps.setMaximum(250)
        self.txt2img_steps_layout = QHBoxLayout()
        self.txt2img_steps_layout.addWidget(self.txt2img_steps_label)
        self.txt2img_steps_layout.addWidget(self.txt2img_steps)

        self.txt2img_cfg_scale_label = QLabel("Cfg scale:")
        self.txt2img_cfg_scale = QDoubleSpinBox()
        self.txt2img_cfg_scale.setMinimum(1.0)
        self.txt2img_cfg_scale.setMaximum(30.0)
        self.txt2img_cfg_scale.setSingleStep(0.5)
        self.txt2img_cfg_scale_layout = QHBoxLayout()
        self.txt2img_cfg_scale_layout.addWidget(self.txt2img_cfg_scale_label)
        self.txt2img_cfg_scale_layout.addWidget(self.txt2img_cfg_scale)

        self.txt2img_batch_count_label = QLabel("Batch count:")
        self.txt2img_batch_count = QSpinBox()
        self.txt2img_batch_count.setMinimum(1)
        self.txt2img_batch_count.setMaximum(250)

        self.txt2img_batch_size_label = QLabel("Batch size:")
        self.txt2img_batch_size = QSpinBox()
        self.txt2img_batch_size.setMinimum(1)
        self.txt2img_batch_size.setMaximum(8)

        self.txt2img_batch_layout = QHBoxLayout()
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_count_label)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_count)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_size_label)
        self.txt2img_batch_layout.addWidget(self.txt2img_batch_size)

        self.txt2img_base_size_label = QLabel("Base size:")
        self.txt2img_base_size = QSpinBox()
        self.txt2img_base_size.setMinimum(64)
        self.txt2img_base_size.setMaximum(2048)
        self.txt2img_base_size.setSingleStep(64)

        self.txt2img_max_size_label = QLabel("Max size:")
        self.txt2img_max_size = QSpinBox()
        self.txt2img_max_size.setMinimum(64)
        self.txt2img_max_size.setMaximum(2048)
        self.txt2img_max_size.setSingleStep(64)

        self.txt2img_size_layout = QHBoxLayout()
        self.txt2img_size_layout.addWidget(self.txt2img_base_size_label)
        self.txt2img_size_layout.addWidget(self.txt2img_base_size)
        self.txt2img_size_layout.addWidget(self.txt2img_max_size_label)
        self.txt2img_size_layout.addWidget(self.txt2img_max_size)

        self.txt2img_seed_label = QLabel("Seed:")
        self.txt2img_seed = QLineEdit()
        self.txt2img_seed.setPlaceholderText("Random")
        self.txt2img_seed_layout = QHBoxLayout()
        self.txt2img_seed_layout.addWidget(self.txt2img_seed_label)
        self.txt2img_seed_layout.addWidget(self.txt2img_seed)

        self.txt2img_use_gfpgan = QCheckBox("Restore faces")
        self.txt2img_use_gfpgan.setTristate(False)

        self.txt2img_tiling = QCheckBox("Enable tiling mode")
        self.txt2img_tiling.setTristate(False)

        self.txt2img_start_button = QPushButton("Apply txt2img")
        self.txt2img_button_layout = QHBoxLayout()
        self.txt2img_button_layout.addWidget(self.txt2img_start_button)

        self.txt2img_layout = QVBoxLayout()
        self.txt2img_layout.addLayout(self.txt2img_prompt)
        self.txt2img_layout.addLayout(self.txt2img_sampler)
        self.txt2img_layout.addLayout(self.txt2img_steps_layout)
        self.txt2img_layout.addLayout(self.txt2img_cfg_scale_layout)
        self.txt2img_layout.addLayout(self.txt2img_batch_layout)
        self.txt2img_layout.addLayout(self.txt2img_size_layout)
        self.txt2img_layout.addLayout(self.txt2img_seed_layout)
        self.txt2img_layout.addWidget(self.txt2img_use_gfpgan)
        self.txt2img_layout.addWidget(self.txt2img_tiling)
        self.txt2img_layout.addLayout(self.txt2img_button_layout)
        self.txt2img_layout.addStretch()

        self.txt2img_widget = QWidget()
        self.txt2img_widget.setLayout(self.txt2img_layout)

    def init_txt2img_interface(self):
        self.txt2img_prompt.cfg_init()
        self.txt2img_sampler.cfg_init()
        self.txt2img_steps.setValue(script.cfg("txt2img_steps", int))
        self.txt2img_cfg_scale.setValue(script.cfg("txt2img_cfg_scale", float))
        self.txt2img_batch_count.setValue(script.cfg("txt2img_batch_count", int))
        self.txt2img_batch_size.setValue(script.cfg("txt2img_batch_size", int))
        self.txt2img_base_size.setValue(script.cfg("txt2img_base_size", int))
        self.txt2img_max_size.setValue(script.cfg("txt2img_max_size", int))
        self.txt2img_seed.setText(script.cfg("txt2img_seed", str))
        self.txt2img_use_gfpgan.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("txt2img_use_gfpgan", bool)
            else Qt.CheckState.Unchecked
        )
        self.txt2img_tiling.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("txt2img_tiling", bool)
            else Qt.CheckState.Unchecked
        )

    def connect_txt2img_interface(self):
        self.txt2img_prompt.cfg_connect()
        self.txt2img_sampler.cfg_connect()
        self.txt2img_steps.valueChanged.connect(
            partial(script.set_cfg, "txt2img_steps")
        )
        self.txt2img_cfg_scale.valueChanged.connect(
            partial(script.set_cfg, "txt2img_cfg_scale")
        )
        self.txt2img_batch_count.valueChanged.connect(
            partial(script.set_cfg, "txt2img_batch_count")
        )
        self.txt2img_batch_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_batch_size")
        )
        self.txt2img_base_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_base_size")
        )
        self.txt2img_max_size.valueChanged.connect(
            partial(script.set_cfg, "txt2img_max_size")
        )
        self.txt2img_seed.textChanged.connect(partial(script.set_cfg, "txt2img_seed"))
        self.txt2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "txt2img_use_gfpgan")
        )
        self.txt2img_tiling.toggled.connect(partial(script.set_cfg, "txt2img_tiling"))
        self.txt2img_start_button.released.connect(lambda: script.action_txt2img())

    def create_img2img_interface(self):
        self.img2img_prompt = QPromptLayout(
            script, "img2img_prompt", "img2img_negative_prompt"
        )

        self.img2img_sampler = QComboBoxLayout(
            script, "img2img_sampler_list", "img2img_sampler", label="Sampler:"
        )

        self.img2img_steps_label = QLabel("Steps:")
        self.img2img_steps = QSpinBox()
        self.img2img_steps.setMinimum(1)
        self.img2img_steps.setMaximum(250)
        self.img2img_steps_layout = QHBoxLayout()
        self.img2img_steps_layout.addWidget(self.img2img_steps_label)
        self.img2img_steps_layout.addWidget(self.img2img_steps)

        self.img2img_cfg_scale_label = QLabel("Cfg scale:")
        self.img2img_cfg_scale = QDoubleSpinBox()
        self.img2img_cfg_scale.setMinimum(1.0)
        self.img2img_cfg_scale.setMaximum(30.0)
        self.img2img_cfg_scale.setSingleStep(0.5)
        self.img2img_cfg_scale_layout = QHBoxLayout()
        self.img2img_cfg_scale_layout.addWidget(self.img2img_cfg_scale_label)
        self.img2img_cfg_scale_layout.addWidget(self.img2img_cfg_scale)

        self.img2img_denoising_strength_label = QLabel("Denoising strength:")
        self.img2img_denoising_strength = QDoubleSpinBox()
        self.img2img_denoising_strength.setMinimum(0.0)
        self.img2img_denoising_strength.setMaximum(1.0)
        self.img2img_denoising_strength.setSingleStep(0.01)
        self.img2img_denoising_strength_layout = QHBoxLayout()
        self.img2img_denoising_strength_layout.addWidget(
            self.img2img_denoising_strength_label
        )
        self.img2img_denoising_strength_layout.addWidget(
            self.img2img_denoising_strength
        )

        self.img2img_batch_count_label = QLabel("Batch count:")
        self.img2img_batch_count = QSpinBox()
        self.img2img_batch_count.setMinimum(1)
        self.img2img_batch_count.setMaximum(250)

        self.img2img_batch_size_label = QLabel("Batch size:")
        self.img2img_batch_size = QSpinBox()
        self.img2img_batch_size.setMinimum(1)
        self.img2img_batch_size.setMaximum(8)

        self.img2img_batch_layout = QHBoxLayout()
        self.img2img_batch_layout.addWidget(self.img2img_batch_count_label)
        self.img2img_batch_layout.addWidget(self.img2img_batch_count)
        self.img2img_batch_layout.addWidget(self.img2img_batch_size_label)
        self.img2img_batch_layout.addWidget(self.img2img_batch_size)

        self.img2img_base_size_label = QLabel("Base size:")
        self.img2img_base_size = QSpinBox()
        self.img2img_base_size.setMinimum(64)
        self.img2img_base_size.setMaximum(2048)
        self.img2img_base_size.setSingleStep(64)

        self.img2img_max_size_label = QLabel("Max size:")
        self.img2img_max_size = QSpinBox()
        self.img2img_max_size.setMinimum(64)
        self.img2img_max_size.setMaximum(2048)
        self.img2img_max_size.setSingleStep(64)

        self.img2img_size_layout = QHBoxLayout()
        self.img2img_size_layout.addWidget(self.img2img_base_size_label)
        self.img2img_size_layout.addWidget(self.img2img_base_size)
        self.img2img_size_layout.addWidget(self.img2img_max_size_label)
        self.img2img_size_layout.addWidget(self.img2img_max_size)

        self.img2img_seed_label = QLabel("Seed:")
        self.img2img_seed = QLineEdit()
        self.img2img_seed.setPlaceholderText("Random")
        self.img2img_seed_layout = QHBoxLayout()
        self.img2img_seed_layout.addWidget(self.img2img_seed_label)
        self.img2img_seed_layout.addWidget(self.img2img_seed)

        self.img2img_checkboxes_layout = QHBoxLayout()
        self.img2img_tiling = QCheckBox("Enable tiling mode")
        self.img2img_tiling.setTristate(False)
        self.img2img_invert_mask = QCheckBox("Invert mask")
        self.img2img_invert_mask.setTristate(False)
        self.img2img_checkboxes_layout.addWidget(self.img2img_tiling)
        self.img2img_checkboxes_layout.addWidget(self.img2img_invert_mask)

        self.img2img_use_gfpgan = QCheckBox("Restore faces")
        self.img2img_use_gfpgan.setTristate(False)

        self.img2img_upscaler = QComboBoxLayout(
            script, "upscaler_list", "img2img_upscaler_name", label="Upscaler:"
        )

        self.img2img_start_button = QPushButton("Apply SD img2img")
        self.img2img_upscale_button = QPushButton("Apply SD upscale")
        self.img2img_inpaint_button = QPushButton("Apply SD inpainting")
        self.img2img_button_layout = QHBoxLayout()
        self.img2img_button_layout.addWidget(self.img2img_start_button)
        # self.img2img_button_layout.addWidget(self.img2img_upscale_button)
        self.img2img_button_layout.addWidget(self.img2img_inpaint_button)

        self.img2img_layout = QVBoxLayout()
        self.img2img_layout.addLayout(self.img2img_prompt)
        self.img2img_layout.addLayout(self.img2img_sampler)
        self.img2img_layout.addLayout(self.img2img_steps_layout)
        self.img2img_layout.addLayout(self.img2img_cfg_scale_layout)
        self.img2img_layout.addLayout(self.img2img_denoising_strength_layout)
        self.img2img_layout.addLayout(self.img2img_batch_layout)
        self.img2img_layout.addLayout(self.img2img_size_layout)
        self.img2img_layout.addLayout(self.img2img_seed_layout)
        self.img2img_layout.addWidget(self.img2img_use_gfpgan)

        self.img2img_layout.addLayout(self.img2img_checkboxes_layout)
        # SD upscale became a script in https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/91bfc71261e160451e89f35a7c0eef66ff98877c
        # self.img2img_layout.addLayout(self.img2img_upscaler)
        self.img2img_layout.addLayout(self.img2img_button_layout)
        self.img2img_layout.addStretch()

        self.img2img_widget = QWidget()
        self.img2img_widget.setLayout(self.img2img_layout)

    def init_img2img_interface(self):
        self.img2img_prompt.cfg_init()
        self.img2img_sampler.cfg_init()
        self.img2img_steps.setValue(script.cfg("img2img_steps", int))
        self.img2img_cfg_scale.setValue(script.cfg("img2img_cfg_scale", float))
        self.img2img_denoising_strength.setValue(
            script.cfg("img2img_denoising_strength", float)
        )
        self.img2img_batch_count.setValue(script.cfg("img2img_batch_count", int))
        self.img2img_batch_size.setValue(script.cfg("img2img_batch_size", int))
        self.img2img_base_size.setValue(script.cfg("img2img_base_size", int))
        self.img2img_max_size.setValue(script.cfg("img2img_max_size", int))
        self.img2img_seed.setText(script.cfg("img2img_seed", str))
        self.img2img_use_gfpgan.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("img2img_use_gfpgan", bool)
            else Qt.CheckState.Unchecked
        )
        self.img2img_tiling.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("img2img_tiling", bool)
            else Qt.CheckState.Unchecked
        )
        self.img2img_invert_mask.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("img2img_invert_mask", bool)
            else Qt.CheckState.Unchecked
        )
        self.img2img_upscaler.cfg_init()

    def connect_img2img_interface(self):
        self.img2img_prompt.cfg_connect()
        self.img2img_sampler.cfg_connect()
        self.img2img_steps.valueChanged.connect(
            partial(script.set_cfg, "img2img_steps")
        )
        self.img2img_cfg_scale.valueChanged.connect(
            partial(script.set_cfg, "img2img_cfg_scale")
        )
        self.img2img_denoising_strength.valueChanged.connect(
            partial(script.set_cfg, "img2img_denoising_strength")
        )
        self.img2img_batch_count.valueChanged.connect(
            partial(script.set_cfg, "img2img_batch_count")
        )
        self.img2img_batch_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_batch_size")
        )
        self.img2img_base_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_base_size")
        )
        self.img2img_max_size.valueChanged.connect(
            partial(script.set_cfg, "img2img_max_size")
        )
        self.img2img_seed.textChanged.connect(partial(script.set_cfg, "img2img_seed"))
        self.img2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "img2img_use_gfpgan")
        )
        self.img2img_tiling.toggled.connect(partial(script.set_cfg, "img2img_tiling"))
        self.img2img_invert_mask.toggled.connect(
            partial(script.set_cfg, "img2img_invert_mask")
        )
        self.img2img_upscaler.cfg_connect()
        self.img2img_start_button.released.connect(lambda: script.action_img2img())
        self.img2img_upscale_button.released.connect(lambda: script.action_sd_upscale())
        self.img2img_inpaint_button.released.connect(lambda: script.action_inpaint())

    def create_upscale_interface(self):
        self.upscale_upscaler = QComboBoxLayout(
            script, "upscaler_list", "upscale_upscaler_name", label="Upscaler:"
        )

        self.upscale_downscale_first = QCheckBox(
            "Downscale image x0.5 before upscaling"
        )
        self.upscale_downscale_first.setTristate(False)

        self.upscale_start_button = QPushButton("Apply upscaler")

        self.upscale_layout = QVBoxLayout()
        self.upscale_layout.addLayout(self.upscale_upscaler)
        self.upscale_layout.addWidget(self.upscale_downscale_first)
        self.upscale_layout.addWidget(self.upscale_start_button)
        self.upscale_layout.addStretch()

        self.upscale_widget = QWidget()
        self.upscale_widget.setLayout(self.upscale_layout)
        return self.upscale_layout

    def init_upscale_interface(self):
        self.upscale_upscaler.cfg_init()
        self.upscale_downscale_first.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("upscale_downscale_first", bool)
            else Qt.CheckState.Unchecked
        )

    def connect_upscale_interface(self):
        self.upscale_upscaler.cfg_connect()
        self.upscale_downscale_first.toggled.connect(
            partial(script.set_cfg, "upscale_downscale_first")
        )
        self.upscale_start_button.released.connect(
            lambda: script.action_simple_upscale()
        )

    def create_config_interface(self):
        self.config_base_url_label = QLabel("Backend url:")
        self.config_base_url = QLineEdit()
        self.config_base_url_reset = QPushButton("Default")
        self.config_base_url_layout = QHBoxLayout()
        self.config_base_url_layout.addWidget(self.config_base_url)
        self.config_base_url_layout.addWidget(self.config_base_url_reset)

        self.config_just_use_yaml = QCheckBox(
            "Use only YAML config, ignore these properties"
        )
        self.config_just_use_yaml.setTristate(False)
        self.config_create_mask_layer = QCheckBox(
            "Create transparency mask layer from selection"
        )
        self.config_create_mask_layer.setTristate(False)
        self.config_delete_temp_files = QCheckBox(
            "Automatically delete temporary image files"
        )
        self.config_delete_temp_files.setTristate(False)
        self.config_fix_aspect_ratio = QCheckBox(
            "Try to fix aspect ratio for selections"
        )
        self.config_fix_aspect_ratio.setTristate(False)
        self.config_only_full_img_tiling = QCheckBox(
            "Allow tiling only with no selection (on full image)"
        )
        self.config_only_full_img_tiling.setTristate(False)

        self.config_sd_model = QComboBoxLayout(
            script, "sd_model_list", "sd_model", label="SD model:"
        )

        self.config_face_restorer_model = QComboBoxLayout(
            script,
            "face_restorer_model_list",
            "face_restorer_model",
            label="Face restorer:",
        )

        self.config_codeformer_weight_label = QLabel(
            "CodeFormer weight (0 - max effect, 1 - min effect)"
        )
        self.config_codeformer_weight = QDoubleSpinBox()
        self.config_codeformer_weight.setMinimum(0.0)
        self.config_codeformer_weight.setMaximum(1.0)
        self.config_codeformer_weight.setSingleStep(0.01)
        self.config_codeformer_weight_layout = QHBoxLayout()
        self.config_codeformer_weight_layout.addWidget(
            self.config_codeformer_weight_label
        )
        self.config_codeformer_weight_layout.addWidget(self.config_codeformer_weight)

        self.config_restore_defaults = QPushButton("Restore Defaults")

        self.config_open_webui_config_label = QLabel(
            '<a href="http://127.0.0.1/">Configure all settings in webUI</a>'
        )
        self.config_weblinks_label = QLabel(
            """
            <a href="http://127.0.0.1:7860/" target="_blank">Configure all settings in webUI</a><br/>
            <a href="https://github.com/Interpause/auto-sd-krita/wiki" target="_blank">Read the guide</a><br/>
            <a href="https://github.com/Interpause/auto-sd-krita/issues" target="_blank">Report bugs or suggest features</a>
            """
        )
        self.config_weblinks_label.setOpenExternalLinks(True)
        self.config_weblinks_label.setWordWrap(True)

        self.config_layout = QVBoxLayout()
        self.config_layout.addWidget(self.config_base_url_label)
        self.config_layout.addLayout(self.config_base_url_layout)
        self.config_layout.addWidget(self.config_just_use_yaml)
        self.config_layout.addWidget(self.config_create_mask_layer)
        self.config_layout.addWidget(self.config_delete_temp_files)
        self.config_layout.addWidget(self.config_fix_aspect_ratio)
        self.config_layout.addWidget(self.config_only_full_img_tiling)
        self.config_layout.addLayout(self.config_sd_model)
        self.config_layout.addLayout(self.config_face_restorer_model)
        self.config_layout.addLayout(self.config_codeformer_weight_layout)
        self.config_layout.addWidget(self.config_restore_defaults)
        self.config_layout.addWidget(self.config_weblinks_label)
        self.config_layout.addStretch()

        self.config_widget = QWidget()
        self.config_widget.setLayout(self.config_layout)

    def init_config_interface(self):
        self.config_base_url.setText(script.cfg("base_url", str))
        self.config_just_use_yaml.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("just_use_yaml", bool)
            else Qt.CheckState.Unchecked
        )
        self.config_create_mask_layer.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("create_mask_layer", bool)
            else Qt.CheckState.Unchecked
        )
        self.config_delete_temp_files.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("delete_temp_files", bool)
            else Qt.CheckState.Unchecked
        )
        self.config_fix_aspect_ratio.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("fix_aspect_ratio", bool)
            else Qt.CheckState.Unchecked
        )
        self.config_only_full_img_tiling.setCheckState(
            Qt.CheckState.Checked
            if script.cfg("only_full_img_tiling", bool)
            else Qt.CheckState.Unchecked
        )
        self.config_sd_model.cfg_init()
        self.config_face_restorer_model.cfg_init()
        self.config_codeformer_weight.setValue(script.cfg("codeformer_weight", float))

    def connect_config_interface(self):
        self.config_base_url.textChanged.connect(partial(script.set_cfg, "base_url"))
        self.config_base_url_reset.released.connect(
            lambda: self.config_base_url.setText(default_url)
        )
        self.config_just_use_yaml.toggled.connect(
            partial(script.set_cfg, "just_use_yaml")
        )
        self.config_create_mask_layer.toggled.connect(
            partial(script.set_cfg, "create_mask_layer")
        )
        self.config_delete_temp_files.toggled.connect(
            partial(script.set_cfg, "delete_temp_files")
        )
        self.config_fix_aspect_ratio.toggled.connect(
            partial(script.set_cfg, "fix_aspect_ratio")
        )
        self.config_only_full_img_tiling.toggled.connect(
            partial(script.set_cfg, "only_full_img_tiling")
        )
        self.config_sd_model.cfg_connect()
        self.config_face_restorer_model.cfg_connect()
        self.config_codeformer_weight.valueChanged.connect(
            partial(script.set_cfg, "codeformer_weight")
        )
        self.config_restore_defaults.released.connect(lambda: self.restore_defaults())

    def restore_defaults(self):
        script.restore_defaults()
        self.update_interfaces()

    def update_remote_config(self):
        script.update_config()
        self.update_interfaces()

    def update_interfaces(self):
        self.init_txt2img_interface()
        self.init_img2img_interface()
        self.init_upscale_interface()
        self.init_config_interface()

    def canvasChanged(self, canvas):
        pass
