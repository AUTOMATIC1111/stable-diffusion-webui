from .krita_diff import *

from functools import partial
import re


# Actions for Hotkeys
class MyExtension(Extension):

    def __init__(self, parent):
        # This is initialising the parent, always important when subclassing.
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        txt2img_action = window.createAction("txt2img", "Apply txt2img transform", "tools/scripts")
        txt2img_action.triggered.connect(
            lambda: script.action_txt2img()
        )
        img2img_action = window.createAction("img2img", "Apply img2img transform", "tools/scripts")
        img2img_action.triggered.connect(
            lambda: script.action_img2img()
        )
        upscale_x_action = window.createAction("img2img_upscale", "Apply img2img upscale transform", "tools/scripts")
        upscale_x_action.triggered.connect(
            lambda: script.action_sd_upscale()
        )
        upscale_x_action = window.createAction("img2img_inpaint", "Apply img2img inpaint transform", "tools/scripts")
        upscale_x_action.triggered.connect(
            lambda: script.action_inpaint()
        )
        simple_upscale_action = window.createAction("simple_upscale", "Apply ESRGAN upscaler", "tools/scripts")
        simple_upscale_action.triggered.connect(
            lambda: script.action_simple_upscale()
        )


# Interface

class KritaSDPluginDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SD Plugin")
        self.create_interface()

        try:
            script.update_config()
        except:
            pass

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

        self.tabs = QTabWidget()
        self.tabs.addTab(self.txt2img_widget, "Txt2Img")
        self.tabs.addTab(self.img2img_widget, "Img2Img")
        self.tabs.addTab(self.upscale_widget, "Upscale")
        self.tabs.addTab(self.config_widget, "Config")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout)

    def create_txt2img_interface(self):
        self.txt2img_prompt_label = QLabel("Prompt:")
        self.txt2img_prompt_text = QPlainTextEdit()
        self.txt2img_prompt_text.setPlaceholderText("krita_config.yaml value will be used")
        self.txt2img_prompt_layout = QHBoxLayout()
        self.txt2img_prompt_layout.addWidget(self.txt2img_prompt_label)
        self.txt2img_prompt_layout.addWidget(self.txt2img_prompt_text)

        self.txt2img_sampler_name_label = QLabel("Sampler:")
        self.txt2img_sampler_name = QComboBox()
        self.txt2img_sampler_name.addItems(samplers)
        self.txt2img_sampler_name_layout = QHBoxLayout()
        self.txt2img_sampler_name_layout.addWidget(self.txt2img_sampler_name_label)
        self.txt2img_sampler_name_layout.addWidget(self.txt2img_sampler_name)

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

        self.txt2img_use_gfpgan = QCheckBox("Enable GFPGAN (may fix faces)")
        self.txt2img_use_gfpgan.setTristate(False)

        self.txt2img_tiling = QCheckBox("Enable tiling mode")
        self.txt2img_tiling.setTristate(False)

        self.txt2img_start_button = QPushButton("Apply txt2img")
        self.txt2img_button_layout = QHBoxLayout()
        self.txt2img_button_layout.addWidget(self.txt2img_start_button)

        self.txt2img_layout = QVBoxLayout()
        self.txt2img_layout.addLayout(self.txt2img_prompt_layout)
        self.txt2img_layout.addLayout(self.txt2img_sampler_name_layout)
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
        self.txt2img_prompt_text.setPlainText(script.cfg('txt2img_prompt', str))
        self.txt2img_sampler_name.setCurrentIndex(script.cfg('txt2img_sampler', int))
        self.txt2img_steps.setValue(script.cfg('txt2img_steps', int))
        self.txt2img_cfg_scale.setValue(script.cfg('txt2img_cfg_scale', float))
        self.txt2img_batch_count.setValue(script.cfg('txt2img_batch_count', int))
        self.txt2img_batch_size.setValue(script.cfg('txt2img_batch_size', int))
        self.txt2img_base_size.setValue(script.cfg('txt2img_base_size', int))
        self.txt2img_max_size.setValue(script.cfg('txt2img_max_size', int))
        self.txt2img_seed.setText(script.cfg('txt2img_seed', str))
        self.txt2img_use_gfpgan.setCheckState(
            Qt.CheckState.Checked if script.cfg('txt2img_use_gfpgan', bool) else Qt.CheckState.Unchecked)
        self.txt2img_tiling.setCheckState(
            Qt.CheckState.Checked if script.cfg('txt2img_tiling', bool) else Qt.CheckState.Unchecked)

    def connect_txt2img_interface(self):
        self.txt2img_prompt_text.textChanged.connect(
            lambda: script.set_cfg("txt2img_prompt", self.txt2img_prompt_text.toPlainText())
        )
        self.txt2img_sampler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "txt2img_sampler")
        )
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
        self.txt2img_seed.textChanged.connect(
            partial(script.set_cfg, "txt2img_seed")
        )
        self.txt2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "txt2img_use_gfpgan")
        )
        self.txt2img_tiling.toggled.connect(
            partial(script.set_cfg, "txt2img_tiling")
        )
        self.txt2img_start_button.released.connect(
            lambda: script.action_txt2img()
        )

    def create_img2img_interface(self):
        self.img2img_prompt_label = QLabel("Prompt:")
        self.img2img_prompt_text = QPlainTextEdit()
        self.img2img_prompt_text.setPlaceholderText("krita_config.yaml value will be used")
        self.img2img_prompt_layout = QHBoxLayout()
        self.img2img_prompt_layout.addWidget(self.img2img_prompt_label)
        self.img2img_prompt_layout.addWidget(self.img2img_prompt_text)

        self.img2img_sampler_name_label = QLabel("Sampler:")
        self.img2img_sampler_name = QComboBox()
        self.img2img_sampler_name.addItems(samplers_img2img)
        self.img2img_sampler_name_layout = QHBoxLayout()
        self.img2img_sampler_name_layout.addWidget(self.img2img_sampler_name_label)
        self.img2img_sampler_name_layout.addWidget(self.img2img_sampler_name)

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
        self.img2img_denoising_strength_layout.addWidget(self.img2img_denoising_strength_label)
        self.img2img_denoising_strength_layout.addWidget(self.img2img_denoising_strength)

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

        self.img2img_tiling = QCheckBox("Enable tiling mode")
        self.img2img_tiling.setTristate(False)

        self.img2img_use_gfpgan = QCheckBox("Enable GFPGAN (may fix faces)")
        self.img2img_use_gfpgan.setTristate(False)

        self.img2img_upscaler_name_label = QLabel("Prescaler for SD upscale:")
        self.img2img_upscaler_name = QComboBox()
        self.img2img_upscaler_name.addItems(upscalers)
        self.img2img_upscaler_name_renew = QPushButton("Update list")
        self.img2img_upscaler_name_layout = QHBoxLayout()
        self.img2img_upscaler_name_layout.addWidget(self.img2img_upscaler_name_label)
        self.img2img_upscaler_name_layout.addWidget(self.img2img_upscaler_name)
        self.img2img_upscaler_name_layout.addWidget(self.img2img_upscaler_name_renew)

        self.img2img_start_button = QPushButton("Apply SD img2img")
        self.img2img_upscale_button = QPushButton("Apply SD upscale")
        self.img2img_inpaint_button = QPushButton("Apply SD inpainting")
        self.img2img_button_layout = QHBoxLayout()
        self.img2img_button_layout.addWidget(self.img2img_start_button)
        self.img2img_button_layout.addWidget(self.img2img_upscale_button)
        self.img2img_button_layout.addWidget(self.img2img_inpaint_button)

        self.img2img_layout = QVBoxLayout()
        self.img2img_layout.addLayout(self.img2img_prompt_layout)
        self.img2img_layout.addLayout(self.img2img_sampler_name_layout)
        self.img2img_layout.addLayout(self.img2img_steps_layout)
        self.img2img_layout.addLayout(self.img2img_cfg_scale_layout)
        self.img2img_layout.addLayout(self.img2img_denoising_strength_layout)
        self.img2img_layout.addLayout(self.img2img_batch_layout)
        self.img2img_layout.addLayout(self.img2img_size_layout)
        self.img2img_layout.addLayout(self.img2img_seed_layout)
        self.img2img_layout.addWidget(self.img2img_use_gfpgan)
        self.img2img_layout.addWidget(self.img2img_tiling)
        self.img2img_layout.addLayout(self.img2img_upscaler_name_layout)
        self.img2img_layout.addLayout(self.img2img_button_layout)
        self.img2img_layout.addStretch()

        self.img2img_widget = QWidget()
        self.img2img_widget.setLayout(self.img2img_layout)

    def init_img2img_interface(self):
        self.img2img_prompt_text.setPlainText(script.cfg('img2img_prompt', str))
        self.img2img_sampler_name.setCurrentIndex(script.cfg('img2img_sampler', int))
        self.img2img_steps.setValue(script.cfg('img2img_steps', int))
        self.img2img_cfg_scale.setValue(script.cfg('img2img_cfg_scale', float))
        self.img2img_denoising_strength.setValue(script.cfg('img2img_denoising_strength', float))
        self.img2img_batch_count.setValue(script.cfg('img2img_batch_count', int))
        self.img2img_batch_size.setValue(script.cfg('img2img_batch_size', int))
        self.img2img_base_size.setValue(script.cfg('img2img_base_size', int))
        self.img2img_max_size.setValue(script.cfg('img2img_max_size', int))
        self.img2img_seed.setText(script.cfg('img2img_seed', str))
        self.img2img_use_gfpgan.setCheckState(
            Qt.CheckState.Checked if script.cfg('img2img_use_gfpgan', bool) else Qt.CheckState.Unchecked)
        self.img2img_tiling.setCheckState(
            Qt.CheckState.Checked if script.cfg('img2img_tiling', bool) else Qt.CheckState.Unchecked)
        self.img2img_upscaler_name.addItems(upscalers[self.img2img_upscaler_name.count():])
        self.img2img_upscaler_name.setCurrentIndex(script.cfg('img2img_upscaler_name', int))

    def connect_img2img_interface(self):
        self.img2img_prompt_text.textChanged.connect(
            lambda: script.set_cfg("img2img_prompt", self.img2img_prompt_text.toPlainText())
        )
        self.img2img_sampler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "img2img_sampler")
        )
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
        self.img2img_seed.textChanged.connect(
            partial(script.set_cfg, "img2img_seed")
        )
        self.img2img_use_gfpgan.toggled.connect(
            partial(script.set_cfg, "img2img_use_gfpgan")
        )
        self.img2img_tiling.toggled.connect(
            partial(script.set_cfg, "img2img_tiling")
        )
        self.img2img_upscaler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "img2img_upscaler_name")
        )
        self.img2img_upscaler_name_renew.released.connect(
            lambda: self.update_upscalers()
        )
        self.img2img_start_button.released.connect(
            lambda: script.action_img2img()
        )
        self.img2img_upscale_button.released.connect(
            lambda: script.action_sd_upscale()
        )
        self.img2img_inpaint_button.released.connect(
            lambda: script.action_inpaint()
        )

    def create_upscale_interface(self):
        self.upscale_upscaler_name_label = QLabel("Upscalers:")
        self.upscale_upscaler_name = QComboBox()
        self.upscale_upscaler_name.addItems(upscalers)
        self.upscale_upscaler_name_renew = QPushButton("Update list")
        self.upscale_upscaler_name_layout = QHBoxLayout()
        self.upscale_upscaler_name_layout.addWidget(self.upscale_upscaler_name_label)
        self.upscale_upscaler_name_layout.addWidget(self.upscale_upscaler_name)
        self.upscale_upscaler_name_layout.addWidget(self.upscale_upscaler_name_renew)

        self.upscale_downscale_first = QCheckBox("Downscale image x0.5 before upscaling")
        self.upscale_downscale_first.setTristate(False)

        self.upscale_start_button = QPushButton("Apply upscaler")

        self.upscale_layout = QVBoxLayout()
        self.upscale_layout.addLayout(self.upscale_upscaler_name_layout)
        self.upscale_layout.addWidget(self.upscale_downscale_first)
        self.upscale_layout.addWidget(self.upscale_start_button)
        self.upscale_layout.addStretch()

        self.upscale_widget = QWidget()
        self.upscale_widget.setLayout(self.upscale_layout)
        return self.upscale_layout

    def init_upscale_interface(self):
        # add loaded upscalers
        self.upscale_upscaler_name.addItems(upscalers[self.upscale_upscaler_name.count():])
        self.upscale_upscaler_name.setCurrentIndex(script.cfg('upscale_upscaler_name', int))
        self.upscale_downscale_first.setCheckState(
            Qt.CheckState.Checked if script.cfg('upscale_downscale_first', bool) else Qt.CheckState.Unchecked)

    def connect_upscale_interface(self):
        self.upscale_upscaler_name.currentIndexChanged.connect(
            partial(script.set_cfg, "upscale_upscaler_name")
        )
        self.upscale_upscaler_name_renew.released.connect(
            lambda: self.update_upscalers()
        )
        self.upscale_downscale_first.toggled.connect(
            partial(script.set_cfg, "upscale_downscale_first")
        )
        self.upscale_start_button.released.connect(
            lambda: script.action_simple_upscale()
        )

    def create_config_interface(self):
        self.config_base_url_label = QLabel("Backend url (only local now):")
        self.config_base_url = QLineEdit()
        self.config_base_url_reset = QPushButton("Default")
        self.config_base_url_layout = QHBoxLayout()
        self.config_base_url_layout.addWidget(self.config_base_url)
        self.config_base_url_layout.addWidget(self.config_base_url_reset)

        self.config_just_use_yaml = QCheckBox("Use only YAML config, ignore these properties")
        self.config_just_use_yaml.setTristate(False)
        self.config_create_mask_layer = QCheckBox("Create transparency mask layer from selection")
        self.config_create_mask_layer.setTristate(False)
        self.config_delete_temp_files = QCheckBox("Automatically delete temporary image files")
        self.config_delete_temp_files.setTristate(False)
        self.config_fix_aspect_ratio = QCheckBox("Try to fix aspect ratio for selections")
        self.config_fix_aspect_ratio.setTristate(False)
        self.config_only_full_img_tiling = QCheckBox("Allow tiling only with no selection (on full image)")
        self.config_only_full_img_tiling.setTristate(False)

        self.config_restore_defaults = QPushButton("Restore Defaults")

        self.config_layout = QVBoxLayout()
        self.config_layout.addWidget(self.config_base_url_label)
        self.config_layout.addLayout(self.config_base_url_layout)
        self.config_layout.addWidget(self.config_just_use_yaml)
        self.config_layout.addWidget(self.config_create_mask_layer)
        self.config_layout.addWidget(self.config_delete_temp_files)
        self.config_layout.addWidget(self.config_fix_aspect_ratio)
        self.config_layout.addWidget(self.config_only_full_img_tiling)
        self.config_layout.addWidget(self.config_restore_defaults)
        self.config_layout.addStretch()

        self.config_widget = QWidget()
        self.config_widget.setLayout(self.config_layout)

    def init_config_interface(self):
        self.config_base_url.setText(script.cfg('base_url', str))
        self.config_just_use_yaml.setCheckState(
            Qt.CheckState.Checked if script.cfg('just_use_yaml', bool) else Qt.CheckState.Unchecked)
        self.config_create_mask_layer.setCheckState(
            Qt.CheckState.Checked if script.cfg('create_mask_layer', bool) else Qt.CheckState.Unchecked)
        self.config_delete_temp_files.setCheckState(
            Qt.CheckState.Checked if script.cfg('delete_temp_files', bool) else Qt.CheckState.Unchecked)
        self.config_fix_aspect_ratio.setCheckState(
            Qt.CheckState.Checked if script.cfg('fix_aspect_ratio', bool) else Qt.CheckState.Unchecked)
        self.config_only_full_img_tiling.setCheckState(
            Qt.CheckState.Checked if script.cfg('only_full_img_tiling', bool) else Qt.CheckState.Unchecked)

    def connect_config_interface(self):
        self.config_base_url.textChanged.connect(
            partial(script.set_cfg, "base_url")
        )
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
        self.config_restore_defaults.released.connect(
            lambda: self.restore_defaults()
        )

    def restore_defaults(self):
        script.restore_defaults()
        self.init_txt2img_interface()
        self.init_img2img_interface()
        self.init_upscale_interface()
        self.init_config_interface()

    def update_upscalers(self):
        script.update_config()
        self.init_img2img_interface()
        self.init_upscale_interface()

    def canvasChanged(self, canvas):
        pass


# And add the extension to Krita's list of extensions:
Krita.instance().addExtension(MyExtension(Krita.instance()))
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("krita_diff", DockWidgetFactoryBase.DockRight, KritaSDPluginDocker))
