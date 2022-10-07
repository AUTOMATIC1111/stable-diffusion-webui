from functools import partial

from krita import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .script import *
from .widgets import QComboBoxLayout, QLineEditLayout, QPromptLayout, QSpinBoxLayout


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
        self.txt2img_prompt_layout = QPromptLayout(
            script, "txt2img_prompt", "txt2img_negative_prompt"
        )
        self.txt2img_sampler_layout = QComboBoxLayout(
            script, "txt2img_sampler_list", "txt2img_sampler", label="Sampler:"
        )

        self.txt2img_steps_layout = QSpinBoxLayout(
            script, "txt2img_steps", label="Steps:", min=1, max=500, step=1
        )

        self.txt2img_cfg_scale_layout = QSpinBoxLayout(
            script, "txt2img_cfg_scale", label="Cfg scale:", min=1.0, max=20.0
        )

        self.txt2img_batch_count_layout = QSpinBoxLayout(
            script, "txt2img_batch_count", label="Batch count:", min=1, max=500, step=1
        )

        self.txt2img_batch_size_layout = QSpinBoxLayout(
            script, "txt2img_batch_size", label="Batch size:", min=1, max=128, step=1
        )

        self.txt2img_batch_layout = QHBoxLayout()
        self.txt2img_batch_layout.addLayout(self.txt2img_batch_count_layout)
        self.txt2img_batch_layout.addLayout(self.txt2img_batch_size_layout)

        self.txt2img_base_size_layout = QSpinBoxLayout(
            script, "txt2img_base_size", "Base size:", min=64, max=8192, step=64
        )

        self.txt2img_max_size_layout = QSpinBoxLayout(
            script, "txt2img_max_size", "Max size:", min=64, max=8192, step=64
        )

        self.txt2img_size_layout = QHBoxLayout()
        self.txt2img_size_layout.addLayout(self.txt2img_base_size_layout)
        self.txt2img_size_layout.addLayout(self.txt2img_max_size_layout)

        self.txt2img_seed_layout = QLineEditLayout(
            script, "txt2img_seed", label="Seed:", placeholder="Random"
        )

        self.txt2img_use_gfpgan = QCheckBox("Restore faces")
        self.txt2img_tiling = QCheckBox("Enable tiling mode")

        self.txt2img_start_button = QPushButton("Apply txt2img")

        self.txt2img_layout = QVBoxLayout()
        self.txt2img_layout.addLayout(self.txt2img_prompt_layout)
        self.txt2img_layout.addLayout(self.txt2img_sampler_layout)
        self.txt2img_layout.addLayout(self.txt2img_steps_layout)
        self.txt2img_layout.addLayout(self.txt2img_cfg_scale_layout)
        self.txt2img_layout.addLayout(self.txt2img_batch_layout)
        self.txt2img_layout.addLayout(self.txt2img_size_layout)
        self.txt2img_layout.addLayout(self.txt2img_seed_layout)
        self.txt2img_layout.addWidget(self.txt2img_use_gfpgan)
        self.txt2img_layout.addWidget(self.txt2img_tiling)
        self.txt2img_layout.addWidget(self.txt2img_start_button)
        self.txt2img_layout.addStretch()

        self.txt2img_widget = QWidget()
        self.txt2img_widget.setLayout(self.txt2img_layout)

    def init_txt2img_interface(self):
        self.txt2img_prompt_layout.cfg_init()
        self.txt2img_sampler_layout.cfg_init()
        self.txt2img_steps_layout.cfg_init()
        self.txt2img_cfg_scale_layout.cfg_init()
        self.txt2img_batch_count_layout.cfg_init()
        self.txt2img_batch_size_layout.cfg_init()
        self.txt2img_base_size_layout.cfg_init()
        self.txt2img_max_size_layout.cfg_init()
        self.txt2img_seed_layout.cfg_init()
        self.txt2img_use_gfpgan.setChecked(script.cfg("txt2img_use_gfpgan", bool))
        self.txt2img_tiling.setChecked(script.cfg("txt2img_tiling", bool))

    def connect_txt2img_interface(self):
        self.txt2img_prompt_layout.cfg_connect()
        self.txt2img_sampler_layout.cfg_connect()
        self.txt2img_steps_layout.cfg_connect()
        self.txt2img_cfg_scale_layout.cfg_connect()
        self.txt2img_batch_count_layout.cfg_connect()
        self.txt2img_batch_size_layout.cfg_connect()
        self.txt2img_base_size_layout.cfg_connect()
        self.txt2img_max_size_layout.cfg_connect()
        self.txt2img_seed_layout.cfg_connect()
        self.txt2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "txt2img_use_gfpgan")
        )
        self.txt2img_tiling.toggled.connect(partial(script.set_cfg, "txt2img_tiling"))
        self.txt2img_start_button.released.connect(lambda: script.action_txt2img())

    def create_img2img_interface(self):
        self.img2img_prompt_layout = QPromptLayout(
            script, "img2img_prompt", "img2img_negative_prompt"
        )

        self.img2img_sampler_layout = QComboBoxLayout(
            script, "img2img_sampler_list", "img2img_sampler", label="Sampler:"
        )

        self.img2img_steps_layout = QSpinBoxLayout(
            script, "img2img_steps", label="Steps:", min=1, max=500, step=1
        )

        self.img2img_cfg_scale_layout = QSpinBoxLayout(
            script, "img2img_cfg_scale", label="Cfg scale:", min=1.0, max=20.0
        )

        self.img2img_denoising_strength_layout = QSpinBoxLayout(
            script, "img2img_denoising_strength", label="Denoising strength:", step=0.01
        )

        self.img2img_batch_count_layout = QSpinBoxLayout(
            script, "img2img_batch_count", label="Batch count:", min=1, max=500, step=1
        )

        self.img2img_batch_size_layout = QSpinBoxLayout(
            script, "img2img_batch_size", label="Batch size:", min=1, max=128, step=1
        )

        self.img2img_batch_layout = QHBoxLayout()
        self.img2img_batch_layout.addLayout(self.img2img_batch_count_layout)
        self.img2img_batch_layout.addLayout(self.img2img_batch_size_layout)

        self.img2img_base_size_layout = QSpinBoxLayout(
            script, "img2img_base_size", "Base size:", min=64, max=8192, step=64
        )

        self.img2img_max_size_layout = QSpinBoxLayout(
            script, "img2img_max_size", "Max size:", min=64, max=8192, step=64
        )

        self.img2img_size_layout = QHBoxLayout()
        self.img2img_size_layout.addLayout(self.img2img_base_size_layout)
        self.img2img_size_layout.addLayout(self.img2img_max_size_layout)

        self.img2img_seed_layout = QLineEditLayout(
            script, "img2img_seed", label="Seed:", placeholder="Random"
        )

        self.img2img_checkboxes_layout = QHBoxLayout()
        self.img2img_tiling = QCheckBox("Enable tiling mode")
        self.img2img_invert_mask = QCheckBox("Invert mask")
        self.img2img_checkboxes_layout.addWidget(self.img2img_tiling)
        self.img2img_checkboxes_layout.addWidget(self.img2img_invert_mask)

        self.img2img_use_gfpgan = QCheckBox("Restore faces")

        self.img2img_upscaler_layout = QComboBoxLayout(
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
        self.img2img_layout.addLayout(self.img2img_prompt_layout)
        self.img2img_layout.addLayout(self.img2img_sampler_layout)
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
        self.img2img_prompt_layout.cfg_init()
        self.img2img_sampler_layout.cfg_init()
        self.img2img_steps_layout.cfg_init()
        self.img2img_cfg_scale_layout.cfg_init()
        self.img2img_denoising_strength_layout.cfg_init()
        self.img2img_batch_count_layout.cfg_init()
        self.img2img_batch_size_layout.cfg_init()
        self.img2img_base_size_layout.cfg_init()
        self.img2img_max_size_layout.cfg_init()
        self.img2img_seed_layout.cfg_init()
        self.img2img_use_gfpgan.setChecked(script.cfg("img2img_use_gfpgan", bool))
        self.img2img_tiling.setChecked(script.cfg("img2img_tiling", bool))
        self.img2img_invert_mask.setChecked(script.cfg("img2img_invert_mask", bool))
        self.img2img_upscaler_layout.cfg_init()

    def connect_img2img_interface(self):
        self.img2img_prompt_layout.cfg_connect()
        self.img2img_sampler_layout.cfg_connect()
        self.img2img_steps_layout.cfg_connect()
        self.img2img_cfg_scale_layout.cfg_connect()
        self.img2img_denoising_strength_layout.cfg_connect()
        self.img2img_batch_count_layout.cfg_connect()
        self.img2img_batch_size_layout.cfg_connect()
        self.img2img_base_size_layout.cfg_connect()
        self.img2img_max_size_layout.cfg_connect()
        self.img2img_seed_layout.cfg_connect()
        self.img2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "img2img_use_gfpgan")
        )
        self.img2img_tiling.toggled.connect(partial(script.set_cfg, "img2img_tiling"))
        self.img2img_invert_mask.toggled.connect(
            partial(script.set_cfg, "img2img_invert_mask")
        )
        self.img2img_upscaler_layout.cfg_connect()
        self.img2img_start_button.released.connect(lambda: script.action_img2img())
        self.img2img_upscale_button.released.connect(lambda: script.action_sd_upscale())
        self.img2img_inpaint_button.released.connect(lambda: script.action_inpaint())

    def create_upscale_interface(self):
        self.upscale_upscaler_layout = QComboBoxLayout(
            script, "upscaler_list", "upscale_upscaler_name", label="Upscaler:"
        )

        self.upscale_downscale_first = QCheckBox(
            "Downscale image x0.5 before upscaling"
        )

        self.upscale_start_button = QPushButton("Apply upscaler")

        self.upscale_layout = QVBoxLayout()
        self.upscale_layout.addLayout(self.upscale_upscaler_layout)
        self.upscale_layout.addWidget(self.upscale_downscale_first)
        self.upscale_layout.addWidget(self.upscale_start_button)
        self.upscale_layout.addStretch()

        self.upscale_widget = QWidget()
        self.upscale_widget.setLayout(self.upscale_layout)
        return self.upscale_layout

    def init_upscale_interface(self):
        self.upscale_upscaler_layout.cfg_init()
        self.upscale_downscale_first.setChecked(
            script.cfg("upscale_downscale_first", bool)
        )

    def connect_upscale_interface(self):
        self.upscale_upscaler_layout.cfg_connect()
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
        self.config_create_mask_layer = QCheckBox(
            "Create transparency mask layer from selection"
        )
        self.config_delete_temp_files = QCheckBox(
            "Automatically delete temporary image files"
        )
        self.config_fix_aspect_ratio = QCheckBox(
            "Try to fix aspect ratio for selections"
        )
        self.config_only_full_img_tiling = QCheckBox(
            "Allow tiling only with no selection (on full image)"
        )

        self.config_sd_model_layout = QComboBoxLayout(
            script, "sd_model_list", "sd_model", label="SD model:"
        )

        self.config_face_restorer_model_layout = QComboBoxLayout(
            script,
            "face_restorer_model_list",
            "face_restorer_model",
            label="Face restorer:",
        )

        self.config_codeformer_weight_layout = QSpinBoxLayout(
            script,
            "codeformer_weight",
            label="CodeFormer weight (max 0, min 1):",
            step=0.01,
        )

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
        self.config_layout.addLayout(self.config_sd_model_layout)
        self.config_layout.addLayout(self.config_face_restorer_model_layout)
        self.config_layout.addLayout(self.config_codeformer_weight_layout)
        self.config_layout.addWidget(self.config_restore_defaults)
        self.config_layout.addWidget(self.config_weblinks_label)
        self.config_layout.addStretch()

        self.config_widget = QWidget()
        self.config_widget.setLayout(self.config_layout)

    def init_config_interface(self):
        self.config_base_url.setText(script.cfg("base_url", str))
        self.config_just_use_yaml.setChecked(script.cfg("just_use_yaml", bool))
        self.config_create_mask_layer.setChecked(script.cfg("create_mask_layer", bool))
        self.config_delete_temp_files.setChecked(script.cfg("delete_temp_files", bool))
        self.config_fix_aspect_ratio.setChecked(script.cfg("fix_aspect_ratio", bool))
        self.config_only_full_img_tiling.setChecked(
            script.cfg("only_full_img_tiling", bool)
        )
        self.config_sd_model_layout.cfg_init()
        self.config_face_restorer_model_layout.cfg_init()
        self.config_codeformer_weight_layout.cfg_init()

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
        self.config_sd_model_layout.cfg_connect()
        self.config_face_restorer_model_layout.cfg_connect()
        self.config_codeformer_weight_layout.cfg_connect()
        self.config_restore_defaults.released.connect(lambda: self.restore_defaults())

    def restore_defaults(self):
        script.restore_defaults()
        script.update_config()
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
