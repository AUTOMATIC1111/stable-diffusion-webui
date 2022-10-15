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
from modules import sd_samplers, sd_models
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
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
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

options_templates.update(options_section(('saving-images', "Saving images/grids|保存图片/网格"), {
    "samples_save": OptionInfo(True, "Always save all generated images|始终保存所有生成的图像"),
    "samples_format": OptionInfo('png', 'File format for images|图像的文件格式'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern|图像文件名样式"),

    "grid_save": OptionInfo(True, "Always save all generated image grids|始终保存所有生成的图像网格"),
    "grid_format": OptionInfo('png', 'File format for grids|网格的文件格式'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid|保存网格时将扩展信息（随机种、提示词）添加到文件名"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture|不保存包含一张图片的网格"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size|网格行计数；使用-1表示自动检测，使用0表示与批大小相同", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files|将有关生成参数的文本信息作为块保存到png文件"),
    "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters.|使用生成参数在每个图像旁边创建一个文本文件"),
    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration.|在进行面部复原之前保存图像副本"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images|保存jpg图像的质量", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG|如果PNG图像大于4MB或任何尺寸大于4000，请缩小比例并将副本保存为JPG"),

    "use_original_name_batch": OptionInfo(False, "Use original name for output filename during batch process in extras tab|在附加选项卡中的批处理过程中，对输出文件名使用原始名称"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image|使用“保存”按钮时，仅保存单个选定图像"),
    "do_not_add_watermark": OptionInfo(False, "Do not add watermark to images|不向图像添加水印"),
}))

options_templates.update(options_section(('saving-paths', "Paths for saving|保存路径"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below|图像输出目录；如果为空，则默认为以下三个目录", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images|文本到图像的输出目录', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images|图像到图像的输出目录', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab|图像到附加的输出目录', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below|网格输出目录；如果为空，则默认为以下两个目录", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids|文本到图像网格的输出目录', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids|图像到图像网格的输出目录', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button|使用“保存”按钮保存图像的目录", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory|保存到目录"), {
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory|将图像保存到子目录"),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory|将网格保存到子目录"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory|使用“保存”按钮时，将图像保存到子目录"),
    "directories_filename_pattern": OptionInfo("", "Directory name pattern|目录名称模式"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern|最大[prompt_words] 样式提示词数量", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}),
}))

options_templates.update(options_section(('upscaling', "Upscaling|放大"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.|ESRGAN放大器的平铺尺寸，0=不进行平铺处理", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.|ESRGAN放大器的平铺重叠，低数值=可见接缝", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN x4+", "R-ESRGAN x4+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)|选择要在web UI中显示的Real ESRGAN模型。（需要重新启动）", gr.CheckboxGroup, lambda: {"choices": realesrgan_models_names()}),
    "SWIN_tile": OptionInfo(192, "Tile size for all SwinIR.|所有SwinIR放大器的平铺尺寸", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}),
    "SWIN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.|SwinIR放大器的平铺重叠，低数值=可见接缝", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "ldsr_steps": OptionInfo(100, "LDSR processing steps. Lower = faster|LDSR处理步数。更低=更快", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img|图像到图像的放大器", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
}))

options_templates.update(options_section(('face-restoration', "Face restoration|脸部修复"), {
    "face_restoration_model": OptionInfo(None, "Face restoration model|脸部修复模型", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect|CodeFormer权重参数；0=最大效果；1=最小影响", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing|处理后将人脸恢复模型从VRAM移动到RAM"),
}))

options_templates.update(options_section(('system', "System|系统"), {
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.|生成期间每秒VRAM使用情况轮询。设置为0以禁用。", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output|始终将所有生成信息打印到标准输出"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job.|向控制台添加第二个进度条，显示整个作业的进度"),
}))

options_templates.update(options_section(('training', "TrainingTraining|训练"), {
    "unload_models_when_training": OptionInfo(False, "nload VAE and CLIP form VRAM when training|训练时从VRAM中卸载VAE和CLIP"),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex|文件名词条正则"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string|文件名加入字符串"),
    "training_image_repeats_per_epoch": OptionInfo(100, "Number of repeats for a single input image per epoch; used only for displaying epoch number|每个纪元单个输入图像的重复次数；仅用于显示纪元编号", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
}))

options_templates.update(options_section(('sd', "Stable Diffusion|稳定扩散"), {
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint|稳定扩散检查点", gr.Dropdown, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, refresh=sd_models.list_models),
    "sd_hypernetwork": OptionInfo("None", "Hypernetwork", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
    "sd_hypernetwork_strength": OptionInfo(1.0, "Hypernetwork strength超网格强度", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors.|对图像到图像结果应用颜色校正以匹配原始颜色。"),
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results|在对img2img图像到图像结果应用颜色校正之前保存图像副本"),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising).|使用图像到图像，精确执行滑块指定的步数（通常，您可以用更少的降噪来减少步数）"),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply.|在K-diffusion采样器中启用量化，以获得更清晰的结果。这可能会改变现有随机种。需要重新启动才能应用"),
    "enable_emphasis": OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention|强调：使用（文本）使模型更加关注文本，而使用[文本]使其更加关注"),
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds.|使用旧的强调实现。可以用来传递旧随机种。"),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image|使K-diffusion采样器批量生成与生成单个图像时相同的图像"),
    "comma_padding_backtrack": OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens|当使用超过75个标记时，通过从n个标记中的最后一个逗号填充来提高一致性", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "filter_nsfw": OptionInfo(False, "Filter NSFW content|过滤NSFW（可能社死）内容"),
    'CLIP_stop_at_last_layers': OptionInfo(1, "Stop At last layers of CLIP model|停止CLIP模型的最后一层", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
    "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button|使用“滚动”按钮时允许随机艺术家选择的类别", gr.CheckboxGroup, {"choices": artist_db.categories()}),
    'quicksettings': OptionInfo("sd_model_checkpoint", "Quicksettings list|快速设置列表"),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options|查询选项(反向训练、反求提示词)"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM|查询：将模型保存在VRAM(显存)中"),
    "interrogate_use_builtin_artists": OptionInfo(True, "Interrogate: use artists from artists.csv|查询：使用artists.csv中的艺术家"),
    "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP|查询：BLIP的num_beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)|查询：最小描述长度（不包括艺术家等）", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length|查询：最大描述长度", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)|查询：文本文件中的最大行数（0=无限制）"),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold|查询：deepbooru分数阈值", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "Interrogate: deepbooru sort alphabetically|查询：deepbooru按字母顺序排序"),
    "deepbooru_use_spaces": OptionInfo(False, "use spaces for tags in deepbooru在deepbooru中为标记使用空格"),
    "deepbooru_escape": OptionInfo(True, "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)deepbooru中的escape（\\）括号（因此它们用作文字括号，而不是强调）"),
}))

options_templates.update(options_section(('ui', "User interface|用户界面"), {
    "show_progressbar": OptionInfo(True, "Show progressbar|显示进度条"),
    "show_progress_every_n_steps": OptionInfo(0, "Show image creation progress every N sampling steps. Set 0 to disable.|每隔N个采样步数显示图像创建进度。将0设置为禁用。", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
    "return_grid": OptionInfo(True, "Show grid in results for web|在web结果中显示图像网格"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in results for web|不在web结果中显示任何图像"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to generation information|将模型哈希添加到生成信息"),
    "add_model_name_to_info": OptionInfo(False, "Add model name to generation information|将模型名称添加到生成信息"),
    "font": OptionInfo("", "Font for image grids that have text|具有文本的图像网格的字体"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer|启用整页图像查看器"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer|在整页图像查看器中显示默认放大的图像"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title.|在窗口标题中显示生成进度"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters|采样器参数"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface (requires restart)|在用户界面中隐藏采样器（需要重新启动）", gr.CheckboxGroup, lambda: {"choices": [x.name for x in sd_samplers.all_samplers]}),
    "eta_ddim": OptionInfo(0.0, "eta (noise multiplier) for DDIM|对DDIM(降噪式扩散隐式模型)使用额噪波倍增", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ancestral": OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers|对ancestral采样器使用噪波倍增", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize|图像到图像DDIM离散化", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn|西格玛混乱系数", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin|西格玛tmin系数",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise|西格玛噪波系数", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta|Eta德尔塔噪波随机种", gr.Number, {"precision": 0}),
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
