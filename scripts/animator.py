#
# Animation Script v0.6
# Inspired by Deforum Notebook
# Must have ffmpeg installed in path.
#
# Explanation of settings:
# Video formats:
#   Create GIF, webM or MP4 file from the series of images. Regardless, .bat files will be created with the right options to make the videos at a later time.
#
# Total Animation Length (s):
#   Total number of seconds to create. Will create fps frames for every second, as you'd expect.
# Framerate:
#   Frames per second to generate.
#
# Denoising Strength:
#   Initial denoising strength value, overrides the value above which is a bit strong for a default. Can be overridden later by keyframes.
# Denoising Decay:
#   Experimental option to enable a half-life decay on the denoising strength. Its value is halved every second.
#
# Zoom Factor (scale/s):
#   Zoom in (>1) or out (<1), at this rate per second. E.g. 2.0 will double size (and crop) every second. Can be overridden later by keyframes.
# X Pixel Shift (pixels/s):
#   Shift the image right (+) or left (-) in pixels per second. Can be overridden later by keyframes.
# Y Pixel Shift (pixels/s):
#   Shift the image down (+) or up (-) in pixels per second. Can be overridden later by keyframes.
#
# Templates: Can be used with keyframes. if not used, img2img prompts or keyframes will be used.
# Positive Prompts:
#   A template for positive prompts that will be added to every keyframe prompt. Allows you to apply a style to an animation.
# Negative Prompts:
#   A template for negative prompts that will be added to every keyframe prompt. Allows you to apply a style to an animation.
#
# Keyframes:
# Format: Time (s) | Desnoise | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Positive Prompts | Negative Prompts
# E.g. Doesn't much, zoom in, move around, zoom out again. expects animation length to be greater than 25s. Prompts blank, templates or the img2img values will be used.
#   0   | 0.4   | 2.0   | 0     | 0     |  | 
#   5   | 0.4   | 1.0   | 250   | 0     |  | 
#   10  | 0.4   | 1.0   | 0     | 250   |  | 
#   15  | 0.4   | 1.0   | -250  | 0     |  | 
#   20  | 0.4   | 1.0   | 0     | -250  |  | 
#   25  | 0.4   | 0.5   | 0     | 0     |  | 


import os, time
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed, process_images
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
import random
import subprocess
import numpy as np
import json
import cv2
import torch

from PIL import Image, ImageFilter
 
def zoom_at2(img, x, y, zoom):
    w, h = img.size
    z = float(zoom)
    #Zoom image
    img2 = img.resize((int(w*z), int(h*z)), Image.Resampling.LANCZOS)    

    #Crate background image
    padding=4    
    resimg = img.resize((w+padding*2, h+padding*2), Image.Resampling.LANCZOS).\
                filter(ImageFilter.GaussianBlur(5)).\
                crop((padding,padding,w+padding,h+padding))
        
    resimg.paste(img2, (int((w - img2.size[0])/2 + x),int((h - img2.size[1])/2 + y)))

    return resimg
 
def opencvtransform(pil_img, angle, translation_x, translation_y, zoom, wrap):
    
    #Convert PIL to OpenCV2 format.
    numpy_image=np.array(pil_img)  
    prev_img_cv2=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    
    #Set up matrices for transformations
    center = (pil_img.size[0] // 2, pil_img.size[1] // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    xform = np.matmul(rot_mat, trans_mat)

    opencv_image = cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE
        )
    
    #Convert OpenCV2 image back to PIL
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)
    
def make_gif(filepath, filename, max_frames, fps, create_vid):
    #Create filenames
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.gif"
    #Build cmd for bat output, local file refs only
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i',  in_filename.replace("%","%%"),
        out_filename
    ]
    #create bat file
    with open(os.path.join(filepath, "makegif.bat"), "w+", encoding="utf-8") as f:
        f.writelines([" ".join(cmd), "\r\n", "pause"])
    #Fix paths for normal output
    cmd[5]= os.path.join(filepath, in_filename)
    cmd[6]= os.path.join(filepath, out_filename)
    #create output if requested
    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
 
def make_webm(filepath, filename, max_frames, fps, create_vid):
 
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.webm"
 
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i',  in_filename.replace("%","%%"),
        '-crf', str(50),
        '-preset', 'veryfast',
        out_filename
    ]
    
    with open(os.path.join(filepath, "makewebm.bat"), "w+", encoding="utf-8") as f:
        f.writelines([" ".join(cmd), "\r\n", "pause"])
    
    cmd[5]= os.path.join(filepath, in_filename)
    cmd[10]= os.path.join(filepath, out_filename)
    
    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
 
def make_mp4(filepath, filename, max_frames, fps, create_vid):
 
    in_filename = f"{str(filename)}_%05d.png"
    out_filename = f"{str(filename)}.mp4"
 
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i',  in_filename.replace("%","%%"),
        '-frames:v', str(max_frames),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        out_filename
    ]
    
    with open(os.path.join(filepath, "makemp4.bat"), "w+", encoding="utf-8") as f:
        f.writelines([" ".join(cmd), "\r\n", "pause"])
    
    cmd[5]= os.path.join(filepath, in_filename)
    cmd[18]= os.path.join(filepath, out_filename)
    
    if create_vid:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
 
class Script(scripts.Script):
 
    def title(self):
        return "Animator"
 
    def show(self, is_img2img):
        return is_img2img
 
    def ui(self, is_img2img):
        
        i1 = gr.HTML("<p style=\"margin-bottom:0.75em\">Render these video formats:</p>")
        with gr.Row():
            vid_gif = gr.Checkbox(label='GIF', value=False)
            vid_mp4 = gr.Checkbox(label='MP4', value=False)
            vid_webm = gr.Checkbox(label='WEBM', value=True)
        
        i2 = gr.HTML("<p style=\"margin-bottom:0.75em\">Animation Parameters</p>")
        with gr.Row():
            totaltime = gr.Textbox(label="Total Animation Length (s)", lines=1, value="10.0")
            fps = gr.Textbox(label="Framerate", lines=1, value="15")
        
        with gr.Row():
            denoising_strength = gr.Slider(label="Denoising Strength, overrides img2img", minimum=0.0, maximum=1.0, step=0.01, value=0.40)
            noise_decay =  gr.Checkbox(label='Denoising Decay', value=False)

        with gr.Row():
            zoom_factor = gr.Textbox(label="Zoom Factor (scale/s)", lines=1, value="1.0")
            x_shift = gr.Textbox(label="X Pixel Shift (pixels/s)", lines=1, value="0")
            y_shift = gr.Textbox(label="Y Pixel Shift (pixels/s)", lines=1, value="0")
        
        i3 = gr.HTML("<p style=\"margin-bottom:0.75em\">Prompt Template, applied to each keyframe below</p>")
        tmpl_pos = gr.Textbox(label="Positive Prompts", lines=1, value="")
        tmpl_neg = gr.Textbox(label="Negative Prompts", lines=1, value="")
        
        i4 = gr.HTML("<p style=\"margin-bottom:0.75em\">Keyframes<br>Format: Time (s) | Desnoise | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Positive Prompts | Negative Prompts</p>")
        prompts = gr.Textbox(label="Keyframes:", lines=5, value="")
        return [i1, i2, i3, i4, totaltime, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos, tmpl_neg, prompts, denoising_strength, x_shift, y_shift, noise_decay]
 
    def run(self, p, i1, i2, i3, i4, totaltime, fps, vid_gif, vid_mp4, vid_webm, zoom_factor, tmpl_pos, tmpl_neg, prompts, denoising_strength, x_shift, y_shift, noise_decay):
      
        outfilename =  time.strftime('%Y%m%d%H%M%S')
        outpath =  os.path.join(p.outpath_samples, outfilename)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        p.do_not_save_samples = True
        p.do_not_save_grid = True
 
        #save settings
        settings_filename = os.path.join(outpath, f"{str(outfilename)}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            sets={}
            sets["Create GIF"]=vid_gif
            sets["Create MP4"]=vid_mp4
            sets["Create WEBM"]=vid_webm
            sets["Total Time"]=totaltime
            sets["FPS"]=fps
            sets["Initial Denoising Strength"]=denoising_strength
            sets["Initial Zoom factor"]=zoom_factor
            sets["Initial X Shift"]=x_shift
            sets["Initial Y Shift"]=y_shift
            sets["Prompt Template, Positive"]=tmpl_pos
            sets["Prompt Template, Negative"]=tmpl_neg
            sets["KeyFrames"]=prompts
            json.dump(dict(sets), f, ensure_ascii=False, indent=4)        
        
        #Build prompt dict of tuples.
        # format of myprompts[framenumber]=("positive prompt","negative prompt")
        myprompts={}
        for myline in prompts.splitlines():
            lineparts = myline.split("|")
            if len(lineparts) < 7:
                continue
            tmpframe = int(float(lineparts[0]) * int(fps))
            myprompts[tmpframe] = (lineparts[1].strip(),lineparts[2].strip(),lineparts[3].strip(),lineparts[4].strip(),lineparts[5].strip(),lineparts[6].strip())
        
        processing.fix_seed(p)
        batch_count = p.n_iter
        
        #Clean up options
        tmpl_pos = str(tmpl_pos).strip()
        tmpl_neg = str(tmpl_neg).strip()
        
        #Save extra parameters for the UI
        p.extra_generation_params = {
            "Total Time (s)": totaltime, 
            "FPS": fps, 
            "Create GIF": vid_gif, 
            "Create MP4": vid_mp4, 
            "Create WEBM": vid_webm, 
            "Prompt Template Positive": tmpl_pos, 
            "Prompt Template Negative": tmpl_neg, 
            "Initial Zoom Factor": zoom_factor, 
            "Initial Denoising Strength": denoising_strength, 
            "Denoise Decay": noise_decay,
            "Initial X Pixel Shift": x_shift, 
            "Initial Y Pixel Shift": y_shift,
            "Keyframe Data": prompts, 
        }
 
        #Check prompts. If no prompt given, but templates exist, set them.
        if len(p.prompt.strip(",").strip()) == 0:           p.prompt =          tmpl_pos
        if len(p.negative_prompt.strip(",").strip()) == 0:  p.negative_prompt = tmpl_neg
        
        if p.init_images[0] is None:
            a = np.random.rand(p.width, p.height, 3)*255
            p.init_images.append(Image.fromarray(a.astype('uint8')).convert('RGB'))
        
        p.batch_size = 1
        p.n_iter = 1
        p.denoising_strength = denoising_strength 
        #For half life, or 0.5x every second, formula:
        # decay_mult =  1/(2^(1/FPS))
        decay_mult = 1 / (2 ** (1 / int(fps)))
        #Zoom FPS scaler = zoom ^ (1/FPS)
        zoom_factor = float(zoom_factor) ** (1/float(fps))
        
        output_images, info = None, None
        initial_seed = None
        initial_info = None
 
        grids = []
        all_images = []
        
        loops = int(fps) * float(totaltime)
        state.job_count = int(loops) * batch_count
 
        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]
 
        x_xhift_cumulitive = 0
        y_shift_cumulitive = 0
        x_shift_perframe = float(x_shift) / float(fps)
        y_shift_perframe = float(y_shift) / float(fps)

        for i in range(int(loops)):
 
            if state.interrupted:
                #Interrupt button pressed in WebUI
                break
            
            #Process Keyframes
            if i in myprompts:
                # Desnoise | Zoom | X Shift | Y shift | Positive Prompts | Negative Prompts | Seed
                print(f"\r\nKeyframe at {i}: {myprompts[i]}\r\n")
                p.denoising_strength =  float(myprompts[i][0])
                zoom_factor =           float(myprompts[i][1]) ** (1/float(fps))
                x_shift =               float(myprompts[i][2])
                y_shift =               float(myprompts[i][3])

                x_shift_perframe = x_shift / int(fps)
                y_shift_perframe = y_shift / int(fps)
                
                #If not prompt, continue previous prompts
                if len(myprompts[i][4]) > 0: p.prompt =          tmpl_pos + ", " + myprompts[i][4]
                if len(myprompts[i][5]) > 0: p.negative_prompt = tmpl_neg + ", " + myprompts[i][5]
            elif noise_decay:
                p.denoising_strength = p.denoising_strength * decay_mult

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.color_corrections = initial_color_corrections
 
            state.job = f"Iteration {i + 1}/{int(loops)}"
 
            processed = processing.process_images(p)
 
            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info
 

            #Accumulate the pixel shift per frame, incase its < 1
            x_shift_cumulitive = x_xhift_cumulitive + x_shift_perframe
            y_shift_cumulitive = y_shift_cumulitive + y_shift_perframe

            #Manipulate image to be passed to next iteration
            init_img = processed.images[0]
            #p.init_images = [zoom_at2(init_img, int(x_shift_cumulitive), int(y_shift_cumulitive), zoom_factor)]
            p.init_images = [opencvtransform(init_img, 0, int(x_shift_cumulitive),  int(y_shift_cumulitive), zoom_factor, False)]
            
            #Subtract the integer portion we just shifted.
            x_shift_cumulitive = x_xhift_cumulitive - int(x_xhift_cumulitive)
            y_shift_cumulitive = y_shift_cumulitive - int(y_shift_cumulitive)

            p.seed = processed.seed + 1
            
            #Save every seconds worth of frames to the output set displayed in UI
            if (i % int(fps) == 0):
                all_images.append(init_img)
             
            #Save current image to folder manually, with specific name we can iterate over.
            init_img.save(os.path.join(outpath, f"{outfilename}_{i:05}.png"))
 
        #If not interrupted, make requested movies. Otherise the bat files exist.
        make_gif(outpath, outfilename, int(loops), int(fps), vid_gif & (not state.interrupted))
        make_mp4(outpath, outfilename, int(loops), int(fps), vid_mp4 & (not state.interrupted))
        make_webm(outpath, outfilename, int(loops), int(fps), vid_webm & (not state.interrupted))
 
 
        #display(all_images, initial_seed, initial_info)
        print("Video Rendered.\r\n")      
 
        processed = Processed(p, all_images, initial_seed, initial_info)
 
        return processed
