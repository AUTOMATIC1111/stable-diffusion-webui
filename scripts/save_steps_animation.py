import json
import os
import shutil

import gradio as gr
from modules import scripts
from modules.images import save_image
from modules.sd_samplers import KDiffusionSampler, sample_to_image

# configurable section
video_rate = 30
author = 'https://github.com/vladmandic'
cli_template = "ffmpeg -hide_banner -loglevel {loglevel} -hwaccel auto -y -framerate {framerate} -i {inpath}/%5d.jpg -r {videorate} {preset} {minterpolate} {flags} -metadata title='{description}' -metadata description='{info}' -metadata author='stable-diffusion' -metadata album_artist='{author}' '{outfile}'" # note: <https://wiki.multimedia.cx/index.php/FFmpeg_Metadata>
presets = {
    'x264': '-vcodec libx264 -preset medium -crf 23',
    'x265': '-vcodec libx265 -preset faster -crf 28',
    'vpx-vp9': '-vcodec libvpx-vp9 -crf 34 -b:v 0 -deadline realtime -cpu-used 4',
    'aom-av1': '-vcodec libaom-av1 -crf 28 -b:v 0 -usage realtime -cpu-used 8 -pix_fmt yuv444p',
}

# internal state variables
current_step = 0
orig_callback_state = KDiffusionSampler.callback_state


class Script(scripts.Script):
    # script title to show in ui
    def title(self):
        return "Save animation of intermediate steps"


    # is ui visible: process/postprocess triggers for always-visible scripts otherwise use run as entry point
    def show(self, is_img2img):
        return scripts.AlwaysVisible


    # ui components
    def ui(self, is_visible):
        with gr.Accordion("Save animation", open = False, elem_id="save-animation"):
            gr.HTML("""
                <a href="https://github.com/vladmandic/generative-art/tree/main/extensions">
                Creates animation sequence from denoised intermediate steps with video frame interpolation to achieve desired animation duration</a><br>""")
            with gr.Row():
                is_enabled = gr.Checkbox(label = "Script Enabled", value = False)
                codec = gr.Radio(label = 'Codec', choices = ['x264', 'x265', 'vpx-vp9', 'aom-av1'], value = 'x264')
                interpolation = gr.Radio(label = 'Interpolation', choices = ['none', 'mci', 'blend'], value = 'mci')
            with gr.Row():
                duration = gr.Slider(label = "Duration", minimum = 0.5, maximum = 120, step = 0.1, value = 10)
                skip_steps = gr.Slider(label = "Skip steps", minimum = 0, maximum = 100, step = 1, value = 5)
            with gr.Row():
                debug = gr.Checkbox(label = "Debug info", value = False)
                run_incomplete = gr.Checkbox(label = "Run on incomplete", value = True)
                tmp_delete = gr.Checkbox(label = "Delete intermediate", value = True)
                out_create = gr.Checkbox(label = "Create animation", value = True)
            with gr.Row():
                tmp_path = gr.Textbox(label = "Path for intermediate files", lines = 1, value = "intermediate")
                out_path = gr.Textbox(label = "Path for output animation file", lines = 1, value = "animation")

        return [is_enabled, codec, interpolation, duration, skip_steps, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path]


    # runs on each step for always-visible scripts
    def process(self, p, is_enabled, codec, interpolation, duration, skip_steps, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path):
        if is_enabled:
            def callback_state(self, d):
                global current_step
                current_step = d["i"] + 1
                if (skip_steps == 0) or (current_step > skip_steps):
                    image = sample_to_image(samples = d["denoised"], index = 0, approximation = None)
                    inpath = os.path.join(p.outpath_samples, tmp_path)
                    save_image(image, inpath, "", extension = 'jpg', short_filename = True, no_prompt = True) # filename using 00000 format so its easier for ffmpeg sequence parsing
                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)


    # run at the end of sequence for always-visible scripts
    def postprocess(self, p, processed, is_enabled, codec, interpolation, duration, skip_steps, debug, run_incomplete, tmp_delete, out_create, tmp_path, out_path):
        global current_step
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)
        if not is_enabled:
            return
        # callback happened too early, it happens with large number of steps and some samplers or if interrupted
        if vars(processed)['steps'] != current_step:
            print('Save animation warning: postprocess early call', { 'current': current_step, 'target': vars(processed)['steps'] })
            if not run_incomplete:
                return
        # create dictionary with all input and output parameters
        v = vars(processed)
        params = {
            'prompt': v['prompt'],
            'negative': v['negative_prompt'],
            'seed': v['seed'],
            'sampler': v['sampler_name'],
            'cfgscale': v['cfg_scale'],
            'steps': v['steps'],
            'current': current_step,
            'skip': skip_steps,
            'info': v['info'].replace('\n', ' '),
            'model': v['info'].split('Model:')[1].split()[0] if ("Model:" in v['info']) else "unknown", # parse string if model info is present
            'embedding': v['info'].split('Used embeddings:')[1].split()[0] if ("Used embeddings:" in v['info']) else "none",  # parse string if embedding info is present
            'faces': v['face_restoration_model'],
            'timestamp': v['job_timestamp'],
            'inpath': os.path.join(p.outpath_samples, tmp_path),
            'outpath': os.path.join(p.outpath_samples, out_path),
            'codec': 'lib' + codec,
            'duration': duration,
            'interpolation': interpolation,
            'loglevel': 'error',
            'cli': cli_template,
            'framerate': 1.0 * (current_step - skip_steps) / duration,
            'videorate': video_rate,
            'author': author,
            'preset': presets[codec],
            'flags': "-movflags +faststart",
            'ffmpeg': shutil.which("ffmpeg"), # detect if ffmpeg executable is present in path
        }
        if debug:
            params['loglevel'] = 'info'
            print("Save animation params:", json.dumps(params, indent = 2))
        if out_create:
            if not os.path.isdir(params['inpath']) or not os.path.isdir(params['outpath']):
                print('Save animation error: folder not found', params['inpath'], params['outpath'])
                return
            if params['ffmpeg'] is None:
                print("Save animation error: ffmpeg not found:")
                return
            # append conditionals to dictionary
            params['minterpolate'] = "" if (params['interpolation'] == "none") else "-vf minterpolate=mi_mode={mi},fifo".format(mi = params['interpolation'])
            params['outfile'] = os.path.join(params['outpath'], str(params['seed']) + "-" + str(params['prompt'])) + ('.webm' if (params['codec'] == 'libvpx-vp9') else '.mp4')
            params['description'] = "{prompt} | negative {negative} | seed {seed} | sampler {sampler} | cfgscale {cfgscale} | steps {steps} | current {current} | model {model} | embedding {embedding} | faces {faces} | timestamp {timestamp} | interpolation {interpolation}".format(**params)
            print("Save animation creating movie sequence:", params['outfile'])
            cmd = params['cli'].format(**params)
            # actual ffmpeg call
            os.system(cmd)
        if tmp_delete:
            for root, _dirs, files in os.walk(params['inpath']):
                print("Save animation removing {n} files from temp folder: {path}".format(path = root, n = len(files)))
                for file in files:
                    f = os.path.join(root, file)
                    if os.path.isfile(f):
                        os.remove(f)
