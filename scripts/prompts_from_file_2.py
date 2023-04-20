import copy
import math
import os
import random
import sys
import traceback
import shlex

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        lines = []
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]

    return None, "\n".join(lines), gr.update(lines=7)

def on_stop_button_click():
    state.interrupt()
    if state.job_count == 0:
        print("🤖 Beep boop! I'm your idling graphic card. 🚀 It's like a vacation! 😂")
    else:
        print(f'{state.job_no} / {state.job_count} jobs interrupted... Aborting now...')


class Script(scripts.Script):

    def title(self):
        return "Prompts from file or textbox (v2)"

    def ui(self, is_img2img):
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style=\"font-size: 1.2em; padding:.75em 0 .15em 0; opacity: 0.5;\">The script will sequently apply each line of prompt as a list from your input, and run jobs in a loop</p>")
                prompt_txt = gr.Textbox(label="Prompt list (seperate with line break)", lines=3, elem_id=self.elem_id("prompt_txt"), placeholder='Enter your prompt(s) here \n(default seperator is line break)',
                                info='By default, enther something in the box will override text prompt (will keep all other parameters including negative prompt)')
                
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                file = gr.File(label="Upload prompt file", type='binary', elem_id=self.elem_id("file"))

            with gr.Column(scale=2, min_width=200):
                separator_txt = gr.Textbox(label="Custom line separator (Optional)", lines=1, elem_id=self.elem_id("separator_txt"), placeholder='Leave blank for default',
                                           info='If you want to use prompt(s) that has multiple lines with line break(s), you might want to set a custom line seperator instead of line break by default') 
                repetitions_slider = gr.Slider(label="Job repeats", step=1, minimum=1, maximum=200, value=1, elem_id=self.elem_id("repetitions_slider"),
                                               info='This will repeat your jobs n times. By default, each repeat will always start with a random seed.')
                stop_button = gr.Button(label="Stop Loop", value ='Force Interrupt and Abort Jobs', elem_id=self.elem_id("force_stop_button"))
            with gr.Column(scale=1, min_width=100):
                
                checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False, elem_id=self.elem_id("checkbox_iterate"))
                checkbox_iterate_batch = gr.Checkbox(label="When seed is -1, use same random seed for all lines", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
                checkbox_same_repeat_seed = gr.Checkbox(label="Use same seed to start all repeat jobs (only when seed is NOT -1)", value=False, elem_id=self.elem_id("repeat_with_random_seed"))
                checkbox_save_grid = gr.Checkbox(label="Save image grid", value=False, elem_id=self.elem_id("checkbox_save_grid"))
        
        with gr.Row():
            gr.Markdown('''<font size="+1">Notes:</font>
            - You can also override other parameters (including negative prompt, checkpoint model, vae, batch setting... etc.) by using format below:
            *--prompt "a cute cat" --negative_prompt "green" --width 1024 --seed 12345678 ...*
            - Each batch job will execute before the prompt loop, meaning that when the batch size or count is set to a value greater than 1, the script will run a complete batch with the same prompt, then use next prompt to run another batch of jobs.
            - If you have set both the repeater and batch greater than 1, the script will run the jobs in the following order: Firstly a complete batch of jobs with prompt 1, then a complete batch with prompt 2, and so on... then move to the next repeat. For example: 
            *repeat 1: [whole batch with prompt1] -> [whole batch with prompt2]... and then move on to repeat 2: [whole batch with prompt1] -> [whole batch with prompt2]...*
            - Therefore, if you only want to generate repeat job in order: prompt1 prompt2 prompt3... just use the job repeat slider, do not set any batch job.
            ''') 

        # We start at one line. When the text changes, we jump to seven lines, or two lines if no \n.
        # We don't shrink back to 1, because that causes the control to ignore [enter], and it may
        # be unclear to the user that shift-enter is needed.
        prompt_txt.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=3), inputs=[prompt_txt], outputs=[prompt_txt])
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt, prompt_txt])
        stop_button.click(fn=on_stop_button_click)
        

        return [checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid, prompt_txt, separator_txt, repetitions_slider, checkbox_same_repeat_seed, start_button]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid, prompt_txt: str, separator_txt: str, repetitions_slider, checkbox_same_repeat_seed, start_button):
        #seperator determine
        separator = separator_txt.strip() if separator_txt.strip() else "\n"
        lines = [x.strip() for x in prompt_txt.split(separator) if x.strip()]
        lines = [x for x in lines if len(x) > 0]

        if checkbox_save_grid:
            p.do_not_save_grid = False
        else:
            p.do_not_save_grid = True

        job_count_each_repeat = 0
        job_count = 0
        jobs = []

        repetitions = repetitions_slider

        for line in lines:
            if "--" in line:
                try:
                    args = cmdargs(line)
                except Exception:
                    print(f"Error parsing line {line} as commandline:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            job_count_each_repeat += args.get("n_iter", p.n_iter)
            job_count = job_count_each_repeat * repetitions

            jobs.append(args)

        print(f"Starting... Will process {len(lines)} prompts in ", f"{job_count_each_repeat} jobs." if repetitions < 2 else f"{job_count_each_repeat} x {repetitions} = {job_count_each_repeat*repetitions} jobs.")

        state.job_count = job_count
        
        images = []
        all_prompts = []
        infotexts = []

        initial_seed = p.seed
        generated_random_seed = -1

        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = generated_random_seed = int(random.randrange(4294967294))

        for r in range(0, int(repetitions)):
            if state.interrupted:
                print(f"🖼️ We have generated a total of {len(images)} images, using the provided prompts. The process was stoped at {state.job_no} of {state.job_count} jobs.")
                repetitions = 1
                break

            elif repetitions > 1:
                print(f'Total [{len(images)}] images created in {state.job_no}/{state.job_count} jobs. Starting repetition [{r+1}/{repetitions}]...')

                #keep seed for "use same seed" 
                if r>0 and checkbox_same_repeat_seed and initial_seed == -1:
                    if generated_random_seed != -1:
                        p.seed = generated_random_seed
                    else:
                        p.seed = initial_seed
                elif r>0 and not checkbox_same_repeat_seed and generated_random_seed != -1:
                    p.seed = int(random.randrange(4294967294))
                elif r>0 and not checkbox_same_repeat_seed:
                    p.seed = -1

            for n, args in enumerate(jobs):
                if state.interrupted:
                    print(f"[{r}/{repetitions}] repetitions" if state.job_no / state.job_count == r else f"{r - 1} / {repetitions} repetitions completed. Aborting jobs.")
                    break
                else:
                    state.job = f"{state.job_no + 1} out of {state.job_count}"

                    copy_p = copy.copy(p)

                    for k, v in args.items():
                        setattr(copy_p, k, v)
                    proc = process_images(copy_p)

                    if checkbox_iterate and generated_random_seed != -1:
                        p.seed = p.seed + (p.batch_size * p.n_iter)

                    images += proc.images
                    all_prompts += proc.all_prompts
                    infotexts += proc.infotexts
                    

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)