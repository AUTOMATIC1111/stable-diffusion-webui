#
# Animation Script v3.0
# Inspired by Deforum Notebook
# Must have ffmpeg installed in path.
# Poor img2img implentation, will trash images that aren't moving.
#
# See https://github.com/Animator-Anon/Animator

import os, time
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images, sd_models
from modules.processing import Processed, process_images
# from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import random
import subprocess
import numpy as np
import pandas as pd
import json
import cv2
# import torch

from PIL import Image, ImageFilter, ImageDraw, ImageFont


def zoom_at2(img, rot, x, y, zoom):
    w, h = img.size

    # Zoom image
    img2 = img.resize((int(w * zoom), int(h * zoom)), Image.Resampling.LANCZOS)

    # Create background image
    padding = 2
    resimg = addnoise(img.copy(), 0.75).resize((w + padding * 2, h + padding * 2), Image.Resampling.LANCZOS). \
        filter(ImageFilter.GaussianBlur(5)). \
        crop((padding, padding, w + padding, h + padding))

    resimg.paste(img2.rotate(rot), (int((w - img2.size[0]) / 2 + x), int((h - img2.size[1]) / 2 + y)))

    return resimg


def pasteprop(img, props, propfolder):

    img2 = img.convert('RGBA')

    for propname in props:
        #prop_name | prop filename | x pos | y pos | scale | rotation
        propfilename = os.path.join(propfolder.strip(), str(props[propname][1]).strip())
        x = int(props[propname][2])
        y = int(props[propname][3])
        scale = float(props[propname][4])
        rotation = float(props[propname][5])

        if not os.path.exists(propfilename):
            print("Prop: Cannot locate file: " + propfilename)
            return img

        prop = Image.open(propfilename)
        w2, h2 = prop.size
        prop2 = prop.resize((int(w2 * scale), int(h2 * scale)), Image.Resampling.LANCZOS).rotate(rotation, expand=True)
        w3, h3 = prop2.size

        tmplayer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        tmplayer.paste(prop2, (int(x - w3 / 2), int(y - h3 / 2)))
        img2 = Image.alpha_composite(img2, tmplayer)

    return img2.convert("RGB")


def rendertext(img, textblocks):
    pad = 1  # Rounding and edge padding of the bubble background.
    d1 = ImageDraw.Draw(img)
    font_size = 20
    for textname in textblocks:
        # textblock_name | text_prompt | x | y | w | h | back_color | white_color | font_filename
        textprompt = str(textblocks[textname][1]).strip().replace('\\n', '\n')
        x = int(textblocks[textname][2])
        y = int(textblocks[textname][3])
        w = int(textblocks[textname][4])
        h = int(textblocks[textname][5])
        # Try convert text to a tuple (255,255,255) or just leave as text "white"
        try:
            backcolor = eval(textblocks[textname][6].strip())
        except:
            backcolor = textblocks[textname][6].strip()
        try:
            forecolor = eval(textblocks[textname][7].strip())
        except:
            forecolor = textblocks[textname][7].strip()
        font_name = str(textblocks[textname][8]).strip()
        # Auto size the text.
        for fs in range(70):
            myfont = ImageFont.truetype(font_name, fs)
            txtsize = d1.multiline_textbbox((0, 0), textprompt, font=myfont)
            if txtsize[2] - txtsize[0] > (w - pad * 2) or txtsize[3] - txtsize[1] > (h - pad * 2):
                font_size = fs - 1
                break

        myfont = ImageFont.truetype(font_name, font_size)
        #print(f"size:{font_size} loc:{x}, {y} size:{w}, {h}")

        #txtsize = d1.multiline_textbbox((0, 0), textprompt, font=myfont)
        #print(f"txtsize:{txtsize}")

        d1.rounded_rectangle((x, y, x + w, y + h), radius=pad, fill=backcolor)
        d1.multiline_text((x + pad, y + pad), textprompt, fill=forecolor, font=myfont, align='center')

    return img


def addnoise(img, percent):
    # Draw coloured circles randomly over the image. Lame, but for testing.
    # print("Noise function")
    w2, h2 = img.size
    draw = ImageDraw.Draw(img)
    for i in range(int(50 * float(percent))):
        x2 = random.randint(0, w2)
        y2 = random.randint(0, h2)
        s2 = random.randint(0, int(50 * float(percent)))
        pos = (x2, y2, x2 + s2, y2 + s2)
        draw.ellipse(pos, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     outline=(0, 0, 0))

    return img


def make_gif(filepath, filename, fps, create_vid, create_bat):
    # Create filenames
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.gif"
    # Build cmd for bat output, local file refs only
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        out_filename
    ]
    # create bat file
    if create_bat:
        with open(os.path.join(filepath, "makegif.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])
    # Fix paths for normal output
    cmd[5] = os.path.join(filepath, in_filename)
    cmd[6] = os.path.join(filepath, out_filename)
    # create output if requested
    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            print(stderr)
#            raise RuntimeError(stderr)

def make_webm(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.webm"

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-crf', str(50),
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makewebm.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[10] = os.path.join(filepath, out_filename)

    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            print(stderr)
#            raise RuntimeError(stderr)

def make_mp4(filepath, filename, fps, create_vid, create_bat):
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.mp4"

    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makemp4.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[16] = os.path.join(filepath, out_filename)

    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#        stdout, stderr = process.communicate()
#        if process.returncode != 0:
#            print(stderr)
#            raise RuntimeError(stderr)

class Script(scripts.Script):

    def title(self):
        return "Animator v3"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        i1 = gr.HTML("<p style=\"margin-bottom:0.75em\">Render these video formats:</p>")
        with gr.Row():
            vid_gif = gr.Checkbox(label="GIF", value=False)
            vid_mp4 = gr.Checkbox(label="MP4", value=False)
            vid_webm = gr.Checkbox(label="WEBM", value=True)

        i2 = gr.HTML("<p style=\"margin-bottom:0.75em\">Animation Parameters</p>")
        with gr.Row():
            totaltime = gr.Textbox(label="Total Animation Length (s)", lines=1, value="10.0")
            fps = gr.Textbox(label="Framerate", lines=1, value="15")
        with gr.Row():
            add_noise = gr.Checkbox(label="Add_Noise", value=False)
            noise_strength = gr.Slider(label="Noise Strength", minimum=0.0, maximum=1.0, step=0.01, value=0.10)
        with gr.Row():
            noise_decay = gr.Checkbox(label="Denoising_Decay", value=False)
            decay_rate = gr.Slider(label="Denoising Decay Rate", minimum=0.1, maximum=1.9, step=0.01, value=0.50)

        i3 = gr.HTML("<p style=\"margin-bottom:0.75em\">Initial Parameters</p>")
        with gr.Row():
            denoising_strength = gr.Slider(label="Denoising Strength (overrides img2img slider)", minimum=0.0,
                                           maximum=1.0, step=0.01, value=0.40)
        with gr.Row():
            zoom_factor = gr.Textbox(label="Zoom Factor (scale/s)", lines=1, value="1.0")
            x_shift = gr.Textbox(label="X Pixel Shift (pixels/s)", lines=1, value="0")
            y_shift = gr.Textbox(label="Y Pixel Shift (pixels/s)", lines=1, value="0")
            rotation = gr.Textbox(label="Rotation (deg/s)", lines=1, value="0")

        i4 = gr.HTML("<p style=\"margin-bottom:0.75em\">Prompt Template, applied to each keyframe below</p>")
        tmpl_pos = gr.Textbox(label="Positive Prompts", lines=1, value="")
        tmpl_neg = gr.Textbox(label="Negative Prompts", lines=1, value="")

        i5 = gr.HTML("<p style=\"margin-bottom:0.75em\">Props</p>")
        propfolder = gr.Textbox(label="Folder:", lines=1, value="")

        # "Time (S) | command word (verbatim as below) | parameters specified by command word below<br>"
        i6 = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Supported Keyframes:<br>"
            "time_s | prompt | positive_prompts | negative_prompts<br>"
            "time_s | transform | zoom | x_shift | y_shift | rotation<br>"
            "time_s | seed | new_seed_int | delta<br>"
            "time_s | subseed | new_subseed_int | delta<br>"
            "time_s | denoise | denoise_value<br>"
            "time_s | prop | prop_filename | x_pos | y_pos | scale | rotation<br>"
            "time_s | set_text | textblock_name | text_prompt | x | y | w | h | fore_color | back_color | font_name<br>"
            "time_s | clear_text | textblock_name<br>"
            "time_s | prop | prop_name | prop_filename | x pos | y pos | scale | rotation<br>"
            "time_s | set_stamp | stamp_name | stamp_filename | x pos | y pos | scale | rotation<br>"
            "time_s | clear_stamp | stamp_name<br>"
            "time_s | col_set<br>"
            "time_s | col_clear<br>"
            "time_s | model | " + ", ".join(
                sorted([x.model_name for x in sd_models.checkpoints_list.values()])) + "</p>")

        prompts = gr.Textbox(label="Keyframes:", lines=5, value="")
        return [i1, i2, i3, i4, i5, i6, totaltime, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos, tmpl_neg,
                prompts, denoising_strength, x_shift, y_shift, rotation, noise_decay, add_noise, noise_strength,
                decay_rate, propfolder]

    def run(self, p, i1, i2, i3, i4, i5, i6, totaltime, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos,
            tmpl_neg, prompts, denoising_strength, x_shift, y_shift, rotation, noise_decay, add_noise, noise_strength,
            decay_rate, propfolder):

        # Fix variable types, i.e. text boxes giving strings.
        totaltime = float(totaltime)
        fps = float(fps)
        zoom_factor = float(zoom_factor)
        x_shift = float(x_shift)
        y_shift = float(y_shift)
        rotation = float(rotation)
        apply_colour_corrections = True

        seed_delta=1
        subseed_delta=0

        outfilename = time.strftime('%Y%m%d%H%M%S')
        outpath = os.path.join(p.outpath_samples, outfilename)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        # Pandas keyframing, hopefully use this to interpolate a bunch of values.
        variables = {'pos1': np.nan,
                     'neg1': np.nan,
                     'pos2': np.nan,
                     'neg2': np.nan,
                     'prompt': np.nan,
                     'denoise': np.nan,
                     'x_shift': np.nan,
                     'y_shift': np.nan,
                     'zoom': np.nan,
                     'rotation': np.nan}

        df = pd.DataFrame(variables, index=range(int(totaltime * fps)))
        # Preload the dataframe with initial values.
        df.loc[0, ['denoise', 'x_shift', 'y_shift', 'zoom', 'rotation']] = [denoising_strength,
                                                                            x_shift / fps,
                                                                            y_shift / fps,
                                                                            zoom_factor ** (1.0 / fps),
                                                                            rotation / fps]

        # Preprocess the keyframe text block into a dictionary, indexed by frame, containing a list of commands (of tuple).
        keyframes = {}
        myprompts = []

        for prompt in prompts.splitlines():
            promptparts = prompt.split("|")
            if len(promptparts) < 2:
                continue
            tmpframe = int(float(promptparts[0]) * fps)
            tmpcommand = promptparts[1].lower().strip()

            if not tmpframe in keyframes:
                keyframes[tmpframe] = []
            keyframes[tmpframe].append(promptparts[1:])

            if tmpcommand == "transform" and len(promptparts) == 6:
                # Time (s) | transform  | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Rotation (deg/s)
                df.loc[tmpframe, ['x_shift', 'y_shift', 'zoom', 'rotation']] = [float(promptparts[3]) / fps,
                                                                                float(promptparts[4]) / fps,
                                                                                float(promptparts[2]) ** (1.0 / fps),
                                                                                float(promptparts[5]) / fps]
            elif tmpcommand == "denoise" and len(promptparts) == 3:
                # Time (s) | denoise | denoise
                df.loc[tmpframe, ['denoise']] = [float(promptparts[2])]

            elif tmpcommand == "prompt" and len(promptparts) == 4:
                # Time (s) | prompt | Positive Prompts | Negative Prompts
                myprompts.append((tmpframe, promptparts[2].strip().strip(",").strip(),
                                            promptparts[3].strip().strip(",").strip()))

        # Sort list of prompts, and then populate the dataframe in a alternating fashion.
        # need to do this to ensure the prompts flow onto each other correctly.
        myprompts = sorted(myprompts)
        #Special case if no prompts supplied.
        if len(myprompts) == 0:
            df.loc[0, ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = ["", "", "", "", 1.0]
        for x in range(len(myprompts)):
            if x == len(myprompts) - 1:
                pos1 = myprompts[x][1]
                pos2 = myprompts[x][1]
                neg1 = myprompts[x][2]
                neg2 = myprompts[x][2]
            else:
                if x % 2:
                    pos1 = myprompts[x + 1][1]
                    pos2 = myprompts[x][1]
                    neg1 = myprompts[x + 1][2]
                    neg2 = myprompts[x][2]
                else:
                    pos1 = myprompts[x][1]
                    pos2 = myprompts[x + 1][1]
                    neg1 = myprompts[x][2]
                    neg2 = myprompts[x + 1][2]

            df.loc[myprompts[x][0], ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = [pos1, neg1,
                                                                                   pos2, neg2,
                                                                                   1.0 - (x % 2)]

        df = df.interpolate(limit_direction='both')
        df.loc[:, ['pos1', 'neg1']] = df.loc[:, ['pos1', 'neg1']].ffill()
        df.loc[:, ['pos2', 'neg2']] = df.loc[:, ['pos2', 'neg2']].ffill()

        #print(df)
        # Check if templates are filled in. If not, try grab prompts at top (i.e. image sent from png info)
        if len(tmpl_pos.strip()) == 0:
            tmpl_pos = p.prompt
        if len(tmpl_neg.strip()) == 0:
            tmpl_neg = p.negative_prompt

        df['pos_prompt'] = tmpl_pos + ", " + df['pos1'] + ":" + df['prompt'].map(str) + ' AND ' + tmpl_pos + ', ' + \
                           df['pos2'] + ":" + (1.0-df['prompt']).map(str)
        df['neg_prompt'] = tmpl_neg + ", " + df['neg1'] + ":" + df['prompt'].map(str) + ' AND ' + tmpl_neg + ', ' + \
                           df['neg2'] + ":" + (1.0-df['prompt']).map(str)

        csv_filename = os.path.join(outpath, f"{str(outfilename)}_frames.csv")
        df.to_csv(csv_filename)
        #print(df)

        processing.fix_seed(p)
        batch_count = p.n_iter

        # Clean up prompt templates
        tmpl_pos = str(tmpl_pos).strip()
        tmpl_neg = str(tmpl_neg).strip()

        # Save extra parameters for the UI
        p.extra_generation_params = {
            "Create GIF": vid_gif,
            "Create MP4": vid_mp4,
            "Create WEBM": vid_webm,
            "Total Time (s)": totaltime,
            "FPS": fps,
            "Initial Denoising Strength": denoising_strength,
            "Initial Zoom Factor": zoom_factor,
            "Initial X Pixel Shift": x_shift,
            "Initial Y Pixel Shift": y_shift,
            "Rotation": rotation,
            "Add Noise": add_noise,
            "Noise Percentage": noise_strength,
            "Denoise Decay": noise_decay,
            "Denoise Decay Rate": decay_rate,
            "Prop Folder": propfolder,
            "Prompt Template Positive": tmpl_pos,
            "Prompt Template Negative": tmpl_neg,
            "Keyframe Data": prompts,
        }

        # save settings, just dump out the extra_generation dict
        settings_filename = os.path.join(outpath, f"{str(outfilename)}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            json.dump(dict(p.extra_generation_params), f, ensure_ascii=False, indent=4)

        # Check prompts. If no prompt given, but templates exist, set them.
        if len(p.prompt.strip(",").strip()) == 0:           p.prompt = tmpl_pos
        if len(p.negative_prompt.strip(",").strip()) == 0:  p.negative_prompt = tmpl_neg

        # This doesn't work, still some information missing if you don't drop an image into the img2img page.
        # if p.init_images[0] is None:
        #    a = np.random.rand(p.width, p.height, 3) * 255
        #    p.init_images.append(Image.fromarray(a.astype('uint8')).convert('RGB'))

        #Post Processing object dicts
        textblocks = {}
        props = {}
        stamps = {}

        p.batch_size = 1
        p.n_iter = 1
        #p.denoising_strength = denoising_strength
        # For half life, or 0.5x every second, formula:
        # decay_mult =  1/(2^(1/FPS))
        #decay_mult = 1 / (2 ** (float(decay_rate) / fps))
        # Zoom FPS scaler = zoom ^ (1/FPS)
        #zoom_factor = zoom_factor ** (1 / fps)

        # output_images, info = None, None
        initial_seed = None
        initial_info = None

        # grids = []
        all_images = []

        # Make bat files before we start rendering video, so we could run them manually to preview output.
        make_gif(outpath, outfilename, fps, False, True)
        make_mp4(outpath, outfilename, fps, False, True)
        make_webm(outpath, outfilename, fps, False, True)

        frame_count = int(fps * totaltime)
        state.job_count = frame_count * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        x_shift_cumulitive = 0
        y_shift_cumulitive = 0
        x_shift_perframe = x_shift / fps
        y_shift_perframe = y_shift / fps
        rot_perframe = rotation / fps

        # Iterate through range of frames
        for frame_no in range(frame_count):

            if state.interrupted:
                # Interrupt button pressed in WebUI
                break

            apply_prop = False
            prop_details = ()

            # Check if keyframes exists for this frame
            if frame_no in keyframes:
                # Keyframes exist for this frame.
                print(f"\r\nKeyframe at {frame_no}: {keyframes[frame_no]}\r\n")

                for keyframe in keyframes[frame_no]:
                    keyframe_command = keyframe[0].lower().strip()
                    # Check the command, should be first item.
                    if keyframe_command == "seed" and len(keyframe) == 3:
                        # Time (s) | seed | seed | delta
                        p.seed = int(keyframe[1])
                        seed_delta = int(keyframe[2])
                        processing.fix_seed(p)
                    elif keyframe_command == "subseed" and len(keyframe) == 3:
                        # Time (s) | subseed | subseed | delta
                        p.subseed = int(keyframe[1])
                        subseed_delta = int(keyframe[2])
                        processing.fix_seed(p)

                    elif keyframe_command == "model" and len(keyframe) == 2:
                        # Time (s) | model    | model name
                        info = sd_models.get_closet_checkpoint_match(keyframe[1].strip() + ".ckpt")
                        if info is None:
                            raise RuntimeError(f"Unknown checkpoint: {keyframe[1]}")
                        sd_models.reload_model_weights(shared.sd_model, info)

                    elif keyframe_command == "col_set" and len(keyframe) == 1:
                        # Time (s) | col_set
                        apply_colour_corrections = True
                        if frame_no > 0:
                            # Colour correction is set automatically above
                            initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
                    elif keyframe_command == "col_clear" and len(keyframe) == 1:
                        # Time (s) | col_clear
                        apply_colour_corrections = False

                    elif keyframe_command == "prop" and len(keyframe) == 6:
                        # Time (s) | prop | prop_filename | x pos | y pos | scale | rotation
                        # bit of a hack, no prop name is supplied, but same function is used to draw.
                        # so the command is passed in place of prop name, which will be ignored anyway.
                        props[len(props)] = keyframe
                    elif keyframe_command == "set_stamp" and len(keyframe) == 7:
                        # Time (s) | set_stamp | stamp_name | stamp_filename | x pos | y pos | scale | rotation
                        stamps[keyframe[1].strip()] = keyframe[1:]
                    elif keyframe_command == "clear_stamp" and len(keyframe) == 2:
                        # Time (s) | clear_stamp | stamp_name
                        if keyframe[1].strip() in stamps:
                            stamps.pop(keyframe[1].strip())

                    elif keyframe_command == "set_text" and len(keyframe) == 10:
                        # time_s | set_text | textblock_name | text_prompt | x | y | w | h | fore_color | back_color | font_name
                        textblocks[keyframe[1].strip()] = keyframe[1:]
                    elif keyframe_command == "clear_text" and len(keyframe) == 2:
                        # Time (s) | clear_text | textblock_name
                        if keyframe[1].strip() in textblocks:
                            textblocks.pop(keyframe[1].strip())

            #elif noise_decay:
            #    p.denoising_strength = p.denoising_strength * decay_mult

            #Grab data from the dataframe and load it into the appropriate areas
            x_shift_perframe = df.loc[frame_no, ['x_shift']][0]
            y_shift_perframe = df.loc[frame_no, ['y_shift']][0]
            rot_perframe = df.loc[frame_no, ['rotation']][0]
            zoom_factor = df.loc[frame_no, ['zoom']][0]
            p.denoising_strength = df.loc[frame_no, ['denoise']][0]

            p.prompt = str(df.loc[frame_no, ['pos_prompt']][0])
            #print(p.prompt)
            p.negative_prompt = str(df.loc[frame_no, ['neg_prompt']][0])
            #print(p.negative_prompt)

            # Extra processing parameters
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            if apply_colour_corrections:
                p.color_corrections = initial_color_corrections

            state.job = f"Iteration {frame_no + 1}/{frame_count}"

            # Process current frame
            processed = processing.process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            # Accumulate the pixel shift per frame, incase it's < 1
            x_shift_cumulitive = x_shift_cumulitive + x_shift_perframe
            y_shift_cumulitive = y_shift_cumulitive + y_shift_perframe

            # Manipulate image to be passed to next iteration
            init_img = processed.images[0]

            # Translate
            init_img = zoom_at2(init_img, rot_perframe, int(x_shift_cumulitive), int(y_shift_cumulitive), zoom_factor)

            # Props
            if len(props) > 0:
                init_img = pasteprop(init_img, props, propfolder)
                props = {}

            # Noise
            if add_noise:
                # print("Adding Noise!!")
                init_img = addnoise(init_img, noise_strength)

            p.init_images = [init_img]

            # Subtract the integer portion we just shifted.
            x_shift_cumulitive = x_shift_cumulitive - int(x_shift_cumulitive)
            y_shift_cumulitive = y_shift_cumulitive - int(y_shift_cumulitive)

            p.seed = processed.seed + seed_delta
            p.subseed = processed.subseed + subseed_delta

            #Post processing (of saved images only)
            post_processed_image = init_img.copy()
            if len(stamps) > 0:
                post_processed_image = pasteprop(post_processed_image, stamps, propfolder)
            if len(textblocks) > 0:
                post_processed_image = rendertext(post_processed_image, textblocks)

            # Save every seconds worth of frames to the output set displayed in UI
            if (frame_no % int(fps) == 0):
                all_images.append(post_processed_image)

            # Save current image to folder manually, with specific name we can iterate over.
            post_processed_image.save(os.path.join(outpath, f"{outfilename}_{frame_no:05}.png"))

        # If not interrupted, make requested movies. Otherise the bat files exist.
        make_gif(outpath, outfilename, fps, vid_gif & (not state.interrupted), False)
        make_mp4(outpath, outfilename, fps, vid_mp4 & (not state.interrupted), False)
        make_webm(outpath, outfilename, fps, vid_webm & (not state.interrupted), False)

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
