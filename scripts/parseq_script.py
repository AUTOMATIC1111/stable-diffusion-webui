import modules.scripts as scripts
import gradio as gr
import subprocess
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from scripts.parseq_core import Parseq
import logging

class Script(scripts.Script):

    def title(self):
        return "SD Parseq v0.01"  

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_path = gr.Textbox(label="Input video path", placeholder="leave blank for img2img loopback mode", lines=1)
        output_path = gr.Textbox(label="Output video path", placeholder="can reference <timestamp> and <img2img_output_path>", value="<img2img_output_path>/parseq-<timestamp>.mp4", lines=1)
        save_images = gr.Checkbox(False, label="Also save frames as images (uses same dir as output video)")
        dry_run_mode = gr.Checkbox(False, label="Dry run mode: don't invoke Stable Diffusion")
        overlay_metadata = gr.Checkbox(False, label="Overlay parameter metadata onto video")
        info2 = gr.HTML("""<p>Generate your <a style="color:blue;" target='_blank' href='https://sd-parseq.web.app'> parameter script on sd-parseq.web.app</a>, then paste the output below. For more info, see <a style="color:blue;" target='_blank' href="https://github.com/rewbs/sd-parseq">sd-parseq on github</a>.</p>
        <p>Note that Parseq overrides many settings you may have specified above, such as scale, denoising strength, prompts and more.</p><br/>""")
        param_script = gr.Textbox(label="Parameter Script", lines=10)

        return [info2, input_path, output_path, save_images, dry_run_mode, overlay_metadata, param_script]

    def run(self, p, info2, input_path, output_path, save_images, dry_run_mode, overlay_metadata, param_script):
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

        # HACK - ideally scripts could opt in to hooks at various points in the processing loop including file saving.
        logging.info("Overriding color correction option to false in main processing, so that we can control color correction in the script loop.")
        old_cc_opt = opts.img2img_color_correction
        opts.img2img_color_correction = False
        p.color_corrections = None
        
        img2img_default_output_path = opts.outdir_samples or opts.outdir_img2img_samples

        original_input_image_resized = images.resize_image(p.resize_mode, p.init_images[0], p.width, p.height) if p.init_images[0] else None
        try:
             [all_images, info] = Parseq().run(p, input_img=original_input_image_resized, input_path=input_path, output_path=output_path,
                                               save_images=save_images, dry_run_mode=dry_run_mode, overlay_metadata=overlay_metadata, default_output_dir=img2img_default_output_path,
                                               param_script_string=param_script, sd_processor=processing)
             Processed(p, all_images, p.seed, info)
        finally:
            logging.info("Restoring CC option to: %s", old_cc_opt)
            opts.img2img_color_correction = old_cc_opt
            logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARNING)            