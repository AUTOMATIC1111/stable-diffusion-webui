import argparse
import datetime
import json
import os
import sys

import gradio as gr
import tqdm

import modules.artists
import modules.interrogate
import modules.memmon
import modules.sd_models
import modules.styles
import modules.devices as devices
from modules import sd_samplers, sd_models, localization
from modules.hypernetworks import hypernetwork
from modules.paths import models_path, script_path, sd_path

sd_model_file = os.path.join(script_path, 'model.ckpt')
default_sd_model_file = sd_model_file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(sd_path, "configs/stable-diffusion/v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=sd_model_file, help="path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded",)
parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to directory with stable diffusion checkpoints")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default=None)
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-half-vae", action='store_true', help="do not switch the VAE model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default=os.path.join(script_path, 'embeddings'), help="embeddings directory for textual inversion (default: embeddings)")
parser.add_argument("--hypernetwork-dir", type=str, default=os.path.join(models_path, 'hypernetworks'), help="hypernetwork directory")
parser.add_argument("--localizations-dir", type=str, default=os.path.join(script_path, 'localizations'), help="localizations directory")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
parser.add_argument("--ngrok-region", type=str, help="The region in which ngrok should start.", default="us")
parser.add_argument("--codeformer-models-path", type=str, help="Path to directory with codeformer model file(s).", default=os.path.join(models_path, 'Codeformer'))
parser.add_argument("--gfpgan-models-path", type=str, help="Path to directory with GFPGAN model file(s).", default=os.path.join(models_path, 'GFPGAN'))
parser.add_argument("--esrgan-models-path", type=str, help="Path to directory with ESRGAN model file(s).", default=os.path.join(models_path, 'ESRGAN'))
parser.add_argument("--bsrgan-models-path", type=str, help="Path to directory with BSRGAN model file(s).", default=os.path.join(models_path, 'BSRGAN'))
parser.add_argument("--realesrgan-models-path", type=str, help="Path to directory with RealESRGAN model file(s).", default=os.path.join(models_path, 'RealESRGAN'))
parser.add_argument("--scunet-models-path", type=str, help="Path to directory with ScuNET model file(s).", default=os.path.join(models_path, 'ScuNET'))
parser.add_argument("--swinir-models-path", type=str, help="Path to directory with SwinIR model file(s).", default=os.path.join(models_path, 'SwinIR'))
parser.add_argument("--ldsr-models-path", type=str, help="Path to directory with LDSR model file(s).", default=os.path.join(models_path, 'LDSR'))
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument("--force-enable-xformers", action='store_true', help="enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; do not make bug reports if this fails to work")
parser.add_argument("--deepdanbooru", action='store_true', help="enable deepdanbooru interrogator")
parser.add_argument("--opt-split-attention", action='store_true', help="force-enables Doggettx's cross-attention layer optimization. By default, it's on for torch cuda.")
parser.add_argument("--opt-split-attention-invokeai", action='store_true', help="force-enables InvokeAI's cross-attention layer optimization. By default, it's on when cuda is unavailable.")
parser.add_argument("--opt-split-attention-v1", action='store_true', help="enable older version of split attention optimization that does not consume all the VRAM it can find")
parser.add_argument("--disable-opt-split-attention", action='store_true', help="force-disables cross-attention layer optimization")
parser.add_argument("--use-cpu", nargs='+',choices=['all', 'sd', 'interrogate', 'gfpgan', 'bsrgan', 'esrgan', 'scunet', 'codeformer'], help="use CPU as torch device for specified modules", default=[], type=str.lower)
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--show-negative-prompt", action='store_true', help="does not do anything", default=False)
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(script_path, 'ui-config.json'))
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(script_path, 'config.json'))
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--gradio-img2img-tool", type=str, help='gradio image uploader tool: can be either editor for ctopping, or color-sketch for drawing', choices=["color-sketch", "editor"], default="editor")
parser.add_argument("--opt-channelslast", action='store_true', help="change memory type for stable diffusion to channels last")
parser.add_argument("--styles-file", type=str, help="filename to use for styles", default=os.path.join(script_path, 'styles.csv'))
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--use-textbox-seed", action='store_true', help="use textbox for seeds in UI (no up/down, but possible to input long seeds)", default=False)
parser.add_argument("--disable-console-progressbars", action='store_true', help="do not output progressbars to console", default=False)
parser.add_argument("--enable-console-prompts", action='store_true', help="print prompts to console when generating with txt2img and img2img", default=False)
parser.add_argument('--vae-path', type=str, help='Path to Variational Autoencoders model', default=None)
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable checking pytorch models for malicious code", default=False)


cmd_opts = parser.parse_args()
restricted_opts = [
    "samples_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
]

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_bsrgan, devices.device_esrgan, devices.device_scunet, devices.device_codeformer = \
(devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'bsrgan', 'esrgan', 'scunet', 'codeformer'])

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False
config_filename = cmd_opts.ui_settings_file

os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
loaded_hypernetwork = None

def reload_hypernetworks():
    global hypernetworks

    hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
    hypernetwork.load_hypernetwork(opts.sd_hypernetwork)


class State:
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    textinfo = None

    def skip(self):
        self.skipped = True

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0
        
    def get_job_timestamp(self):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # shouldn't this return job_timestamp?


state = State()

artist_db = modules.artists.ArtistsDatabase(os.path.join(script_path, 'artists.csv'))

styles_filename = cmd_opts.styles_file
prompt_styles = modules.styles.StyleDatabase(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []

localization.list_localizations(cmd_opts.localizations_dir)


def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, show_on_main_page=False, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = None
        self.refresh = refresh


def options_section(section_identifier, options_dict):
    for k, v in options_dict.items():
        v.section = section_identifier

    return options_dict


hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

options_templates = {}

options_templates.update(options_section(('saving-images', "保存图片/网格图片（单次生成多张）"), {
    "samples_save": OptionInfo(True, "保存所有生成的照片"),
    "samples_format": OptionInfo('png', '图像文件格式'),
    "samples_filename_pattern": OptionInfo("", "图像文件名格式"),

    "grid_save": OptionInfo(True, "保存所有生成的网格图"),
    "grid_format": OptionInfo('png', '网格的文件格式'),
    "grid_extended_filename": OptionInfo(False, "保存网格时将扩展信息（图像生成种子、关键词语句）添加到文件名"),
    "grid_only_if_multiple": OptionInfo(True, "不保存由张图像组成的网格"),
    "grid_prevent_empty_spots": OptionInfo(False, "防止网格中出现空点（当设置为自动检测时）"),
    "n_rows": OptionInfo(-1, "网格行数；使用-1表示自动检测，使用0表示与批次大小相同。", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo": OptionInfo(True, "将关于生成参数的文本信息保存到png文件中（使用记事本打开可以看到信息）"),
    "save_txt": OptionInfo(False, "在生成图片时，将参数以.txt格式保存到同目录中"),
    "save_images_before_face_restoration": OptionInfo(False, "在面部修复前保存一份图像的副本"),
    "jpeg_quality": OptionInfo(80, "保存jpeg图像的质量", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "export_for_4chan": OptionInfo(True, "如果PNG大于4MB或任何尺寸大于4000,缩小尺寸并保存为JPG"),

    "use_original_name_batch": OptionInfo(False, "在额外标签的批处理过程中,为输出文件名使用原始名称"),
    "save_selected_only": OptionInfo(True, "使用“保存”按钮时,只保存一个选定的图像"),
    "do_not_add_watermark": OptionInfo(False, "不在图像中添加水印"),
}))

options_templates.update(options_section(('saving-paths', "保存路径"), {
    "outdir_samples": OptionInfo("", "图像的输出目录;如果为空,默认为下面的三个目录", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'txt2img图像的输出目录', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'img2img图像的输出目录', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", '额外标签的输出目录', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "网格的输出目录;如果为空,默认为下面的两个目录", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'txt2img网格的输出目录', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'img2img网格的输出目录', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "使用“保存”按钮保存图像的目录", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "保存到目录"), {
    "save_to_dirs": OptionInfo(False, "将图像保存到子目录"),
    "grid_save_to_dirs": OptionInfo(False, "将网格保存到子目录"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "当使用“保存”按钮时,将图像保存到子目录"),
    "directories_filename_pattern": OptionInfo("", "目录命名方式"),
    "directories_max_prompt_words": OptionInfo(8, "使用[]描述时的最大提示词数量 (可能指的是所有类型的括号)", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}),
}))

options_templates.update(options_section(('upscaling', "图像高清化"), {
    "ESRGAN_tile": OptionInfo(192, "ESRGAN图像放大器的图块大小. 0 = 不平铺", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "贴图重叠范围,以ESRGAN图像放大器的像素为单位, 低输入值将导致明显可见接缝", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN x4+", "R-ESRGAN x4+ Anime6B"], "选择要在WebUI中显示的RealESRGAN模型(需要重启)", gr.CheckboxGroup, lambda: {"choices": realesrgan_models_names()}),
    "SWIN_tile": OptionInfo(192, "SwinIR模型的无缝拼接图像尺寸.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}),
    "SWIN_tile_overlap": OptionInfo(8, "贴图重叠范围,以SwinIR图像放大器的像素为单位, 低输入值将导致明显可见接缝", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "ldsr_steps": OptionInfo(100, "LDSR处理次数，次数越低处理越快", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}),
    "upscaler_for_img2img": OptionInfo(None, "对img2img使用图像放大器", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "use_scale_latent_for_hires_fix": OptionInfo(False, "在进行高清修复时，提升图像的质量（翻译存疑"),
}))

options_templates.update(options_section(('face-restoration', "面部修复"), {
    "face_restoration_model": OptionInfo(None, "面部修复模型", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "编码器权重, 越接近0权重越高，范围为0-1", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "处理后将面部修复模型从显存移到内存"),
}))

options_templates.update(options_section(('system', "系统"), {
    "memmon_poll_rate": OptionInfo(8, "在生成时，检测VRAM的占用情况，设置为0禁用.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "始终将所有信息打印到标准输出（后台）"),
    "multiple_tqdm": OptionInfo(True, "在控制台添加第二个进度条，显示批处理的进度."),
}))

options_templates.update(options_section(('training', "训练"), {
    "unload_models_when_training": OptionInfo(False, "在训练时，卸载VAE权重模型和CLIP生成模型"),
    "dataset_filename_word_regex": OptionInfo("", "正则文件名"),
    "dataset_filename_join_string": OptionInfo(" ", "文件名连接的字符串"),
    "training_image_repeats_per_epoch": OptionInfo(1, "每一代训练的单个输入图像的重复次数;仅用于显示训练代数", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "每隔N步保存一个包含loss值的csv文件到日志目录, 0表示禁用"),
}))

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion 存档点模型", gr.Dropdown, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, refresh=sd_models.list_models),
    "sd_checkpoint_cache": OptionInfo(0, "存档点模型缓存在RAM中的大小", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_hypernetwork": OptionInfo("None", "Hypernetwork 模型", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
    "sd_hypernetwork_strength": OptionInfo(1.0, "Hypernetwork 模型强度", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
    "img2img_color_correction": OptionInfo(False, "对img2img生成的结果应用颜色校正来与原始颜色相匹配"),
    "save_images_before_color_correction": OptionInfo(False, "在对img2img生成的结果应用颜色校正之前,保存图像的副本"),
    "img2img_fix_steps": OptionInfo(False, "在img2img时, 解析次数等于设定值（通常解析次数会小于设定值，去噪更少）"),
    "enable_quantization": OptionInfo(False, "在K-diffusion采样器中启动量化来获得更清晰简洁的结果,这可能会改变现有的图像生成种子,需要重新启动才能应用."),
    "enable_emphasis": OptionInfo(True, "描述强调: 使用()增加权重，使用[]降低权重"),
    "use_old_emphasis_implementation": OptionInfo(False, "使用旧版强调实现. 这对于老种子有用."),
    "enable_batch_seeds": OptionInfo(True, "使用K-diffusion采样器批量生成与生成单个图像时相同的图像"),
    "comma_padding_backtrack": OptionInfo(20, "当使用超过75个token时,从n个token内的最后一个逗号开始填充,以提高一致性", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "filter_nsfw": OptionInfo(False, "过滤NSFW（R-18）内容"),
    'CLIP_stop_at_last_layers': OptionInfo(1, "在CLIP模型的最后几层停止", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
    "random_artist_categories": OptionInfo([], "当使用随机关键词按钮时,允许选择随机艺术家类别", gr.CheckboxGroup, {"choices": artist_db.categories()}),
}))

options_templates.update(options_section(('interrogate', "询问设置"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "询问:将模型保存在显存中"),
    "interrogate_use_builtin_artists": OptionInfo(True, "询问:使用artsts.csv中的艺术家"),
    "interrogate_return_ranks": OptionInfo(False, "询问:在结果中包含模型标签匹配的排名(对基于标题的询问器没有影响)"),
    "interrogate_clip_num_beams": OptionInfo(1, "询问:集数数量来自BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "询问:最小描述长度(不包括艺术家等)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "询问:最大描述长度", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP:文本文件中的最大行数(0 =无限制)"),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "询问:deepbooru可信度分数阈值", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "询问:deepbooru分类整理"),
    "deepbooru_use_spaces": OptionInfo(False, "在deepbooru中为标签使用空格"),
    "deepbooru_escape": OptionInfo(True, "deepbooru转义(\\)方括号(这将不影响权重值，只是字面意义上的括号)/escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"),
    }))
options_templates.update(options_section(('ui', "用户界面"), {
    "show_progressbar": OptionInfo(True, "显示进度条"),
    "show_progress_every_n_steps": OptionInfo(0, "每N个采样步数更新图像的生成进度,设置0禁用", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
    "return_grid": OptionInfo(True, "在web中显示网格"),
    "do_not_show_images": OptionInfo(False, "网页不显示任何生成图像的结果"),
    "add_model_hash_to_info": OptionInfo(True, "在生成信息中添加模型哈希"),
    "add_model_name_to_info": OptionInfo(False, "将模型名称添加到生成信息中"),
    "font": OptionInfo("", "具有文本的图像网格的字体"),
    "js_modal_lightbox": OptionInfo(True, "启用整页图像查看界面"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "在整页图像查看界面中默认显示放大的图像"),
    "show_progress_in_title": OptionInfo(True, "在浏览器标题中显示生成进度"),
    'quicksettings': OptionInfo("sd_model_checkpoint", "快速设置列表"),
    'localization': OptionInfo("None", "Localization (requires restart)", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)),
}))

options_templates.update(options_section(('sampler-params', "采样器器参数"), {
    "hide_samplers": OptionInfo([], "在用户界面中隐藏采样工具(需要重新启动)", gr.CheckboxGroup, lambda: {"choices": [x.name for x in sd_samplers.all_samplers]}),
    "eta_ddim": OptionInfo(0.0, "eta(噪声倍增器)用于DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ancestral": OptionInfo(1.0, "eta(噪声倍增器)用于原始采样工具", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM 离散化", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma混合", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma时长",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma噪点", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta噪声种子/Eta noise seed delta", gr.Number, {"precision": 0}),
}))


class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func):
        item = self.data_labels.get(key)
        item.onchange = func

        func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)


opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

sd_upscalers = []

sd_model = None

progress_print_out = sys.stdout


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=state.job_count * state.sampling_steps,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total=new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


total_tqdm = TotalTQDM()

mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()
