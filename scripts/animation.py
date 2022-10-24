import math
import os
from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules import images, devices
from modules.images import get_next_sequence_number, save_image
from modules.processing import process_images, Processed, StableDiffusionProcessingImg2Img
from modules.shared import opts, cmd_opts, state
from modules.prompt_parser import get_multicond_learned_conditioning, ComposableScheduledPromptConditioning, MulticondLearnedConditioning
import modules.sd_samplers
import modules.shared as shared

import code
import torch
import cv2
import math
import numpy as np
import json
# Note: in order for this to work, need the patched class KDiffusionSampler which includes new function set_sigma_start_end_indices 

def int_log2(x):
    return int(math.log2(x))
        
def sample_animation_frames(p, x, sample_func, sample_args, conditioning_from_ratio):
    MIN_SQUARE_DIFF_FACTOR = 0.5 # TODO: want this as a parameter? Increasing this can result in smoother animation.
    DONT_AVERAGE_LAST_LEVEL = False # TODO: want it as parameter? probably not, not so helpful as I thought.
    
    current_samples = x
    previous_conditioning_ratios = [0]
    num_animation_frames = p.num_animation_frames
    num_levels = len(p.levels_sizes)
    x_copy = x
    first_sampling_done = False
    
    for current_level in range(num_levels):
        print(f"animation: level {current_level+1}/{num_levels}")
        print(f"current_samples.shape={current_samples.shape}")
        current_level_size = p.levels_sizes[current_level]
        
        new_sample_inputs = []
        new_sample_inputs_indices = []
        new_sample_ratios = []
        if len(previous_conditioning_ratios) <= 1:
            new_sample_inputs = [current_samples[0]] * current_level_size
            new_sample_inputs_indices = [0] * current_level_size
            new_sample_ratios = [1] * current_level_size
            if current_level_size == 1:
                conditioning_ratios = [0.5]
                if p.img2img_img_is_first_frame:
                    conditioning_ratios = [1] # The first frame should already give us enough in common, condition to the end.
            else:
                if p.img2img_img_is_first_frame:
                    conditioning_ratios = [(i+1) / (current_level_size) for i in range(current_level_size)]
                else:
                    conditioning_ratios = [i / (current_level_size - 1) for i in range(current_level_size)]
        else:
            squared_diffs = torch.sqrt(
                torch.sum(
                    torch.square(
                    torch.diff(current_samples, dim=0)
                    ),
                    list(range(1, len(current_samples.shape))))
            ).cpu()
            min_square_diff = torch.min(squared_diffs)
            squared_diffs -= min_square_diff * MIN_SQUARE_DIFF_FACTOR # Try to ignore differences from random
            squared_diffs_sums = torch.cat((torch.zeros(1), torch.cumsum(squared_diffs, dim=0)))
            squared_diffs_sums = previous_conditioning_ratios[0] + squared_diffs_sums * ( (previous_conditioning_ratios[-1] - previous_conditioning_ratios[0]) / squared_diffs_sums[-1])
            print(f"squared_diffs sums: {list(enumerate(squared_diffs_sums))}")
            
            conditioning_ratios = []
            j = 0
            max_j = len(previous_conditioning_ratios) - 1
            for i in range(current_level_size):
                if p.img2img_img_is_first_frame:
                    wanted_ratio = (i+1) / (current_level_size) # First frame is ratio 0.
                else:
                    wanted_ratio = i / (current_level_size - 1)
                
                while j <= max_j and wanted_ratio >= squared_diffs_sums[j]:
                    j+=1
                #print(f"i={i}, wanted_ratio={wanted_ratio}, j={j}")
                if j <= 0:
                    new_sample_inputs.append(current_samples[0])
                    new_sample_ratios.append(1)
                    conditioning_ratios.append(wanted_ratio)
                elif j > max_j:
                    j = max_j
                    new_sample_inputs.append(current_samples[j])
                    conditioning_ratios.append(wanted_ratio)
                    new_sample_ratios.append(1)
                else:
                    j -= 1
                    #print(f"i={i}, wanted_ratio={wanted_ratio}, j={j}, squared_diffs_sums[{j}]={squared_diffs_sums[j]}, squared_diffs_sums[{j+1}]={squared_diffs_sums[j+1]}")
                    first_ratio = 1 - (wanted_ratio - squared_diffs_sums[j]) / (squared_diffs_sums[j+1] - squared_diffs_sums[j])
                    new_sample_ratios.append(first_ratio)
                    conditioning_ratio = previous_conditioning_ratios[j] * first_ratio + (1-first_ratio) * previous_conditioning_ratios[j+1]
                    conditioning_ratios.append(conditioning_ratio)
                    if (current_level == num_levels - 1 and DONT_AVERAGE_LAST_LEVEL) or p.dont_average_animation: # Don't average in the last level
                        if first_ratio >= 0.5:
                            new_sample = current_samples[j]
                        else:
                            new_sample = current_samples[j+1]
                    else:
                        new_sample = current_samples[j] * first_ratio + (1-first_ratio) * current_samples[j+1]
                    new_sample_inputs.append(new_sample)
                if i == current_level_size - 1:
                    assert(torch.all(torch.eq(new_sample_inputs[-1], current_samples[-1])))
                new_sample_inputs_indices.append(j)
                

        print(f"new samples indices taken from: {new_sample_inputs_indices}")
        print(f"conditioning_ratios: {conditioning_ratios}")
        print(f"[(i, ratio, parent_sample_index, sample_ratio)] = {list(zip(range(current_level_size), conditioning_ratios, new_sample_inputs_indices, new_sample_ratios))}")
        new_sample_inputs = torch.cat([sample[None] for sample in new_sample_inputs])
        sigma_start = p.sigmas_per_level[current_level]
        if current_level == num_levels - 1:
            sigma_end = None
        else:
            sigma_end = p.sigmas_per_level[current_level + 1] + 1 # +1 to sigma_end because of a bug where it always skips the last sigma.
        print(f"animation: level {current_level+1}/{num_levels}, sigma_start={sigma_start}, sigma_end={sigma_end}")
        print(f"new_sample_inputs shape={new_sample_inputs.shape}, current_samples.shape = {current_samples.shape}")
        p.sampler.set_sigma_start_end_indices(sigma_start, sigma_end)
        for new_index in range(current_level_size):
            assert(sigma_start > 0 or not first_sampling_done) # we sample ony once with the first sigma index.
            if state.interrupted:
                print(f"stopped prematurely")
                return current_samples
                
            batch_results = sample_func(x=new_sample_inputs[new_index:new_index+1], conditioning=conditioning_from_ratio(conditioning_ratios[new_index]), **sample_args)
            first_sampling_done = True
            devices.torch_gc()
            if new_index == 0:
                if p.img2img_img_is_first_frame:
                    # Always inject the original image sample into this. 
                    # We're faking a "perfect" denoising by just reducing the noise multiplier.
                    # We still need noise or the images will turn out blurry.
                    # Note that the last sample will be identical to the img2img frame as we multiply by 0 the noise.
                    current_samples = torch.cat((x_copy + sample_args["noise"] * p.sigma_sched[sigma_end or -1], batch_results)) 
                else:
                    current_samples = batch_results
            else:
                current_samples = torch.cat((current_samples, batch_results))
                
            del batch_results
                
        del new_sample_inputs
        devices.torch_gc()
        previous_conditioning_ratios = conditioning_ratios
        if p.img2img_img_is_first_frame:
            previous_conditioning_ratios = [0] + conditioning_ratios
        assert(len(previous_conditioning_ratios) == current_samples.shape[0])
    
    return current_samples

VIDEO_EXTENSION = ".mp4" # Only supported right now.
def save_animation(output_frames, output_path, codec=None, fps = None, animation_duration = 5): # TODO: fps / frames interpolation.
    if fps is None:
        fps = max(1, len(output_frames) // animation_duration) # At least 1 FPS.
    if output_path.endswith("avi") and codec is None: # avi codec isn't supported by gradio, so don't even give an option.
        codec = "XVID"
    elif output_path.endswith("mp4") and codec is None:
        codec = "avc1" # Need to download binaries from cisco for this to work.
    print(f"saving animation to '{output_path}', codec={codec}, fps={fps}, number of frames: {len(output_frames)}")
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*codec), fps, (output_frames[0].width, output_frames[0].height))
    for frame in output_frames:
        cv_im = np.array(frame)
        out.write(cv_im[:,:,::-1])
    out.release()
    
def weight_conditioning_list(conditioning_list, weight):
    return [ComposableScheduledPromptConditioning(schedules=c.schedules, weight=c.weight * weight) for c in conditioning_list]
    
def sample_animation(p, x, conditioning, unconditional_conditioning, steps, is_img2img, noise=None, image_conditioning=None):
    print("sampling animation")
    if type(p.sampler) != modules.sd_samplers.KDiffusionSampler:
        raise NotImplementedError("Animation implemented for KDiffusionSampler only")
    if len(conditioning.batch) != 1:
        raise ValueError("Animation received batch conditioning")
    
    sample_args = {
        'p' : p,
        'unconditional_conditioning' : unconditional_conditioning,
        'steps' : steps,
        'image_conditioning' : image_conditioning
    }
    if is_img2img:
        sample_args['noise'] = noise
        sample_func = p.sampler.sample_img2img
        p.sigma_sched = list(p.sampler.get_sigmas(p, p.fake_steps)[p.fake_steps-p.t_enc-1:])
    else:
        sample_func = p.sampler.sample
        p.img2img_img_is_first_frame = False
        
    print(f"sample_animation: steps={steps}, p.steps={p.steps}")
    conditioning_end = get_multicond_learned_conditioning(shared.sd_model, [p.prompt_end], p.steps)
     
    def conditioning_from_ratio(ratio):
        new_conditioning = [weight_conditioning_list(conditioning.batch[0], 1-ratio) + weight_conditioning_list(conditioning_end.batch[0], ratio)]
        return MulticondLearnedConditioning(conditioning.shape, new_conditioning)
        
    frames = sample_animation_frames(p, x, sample_func, sample_args, conditioning_from_ratio)
    return frames
    
# Hooking sampler.sample and processing.init
# (Sampler isn't initialized in the beggining, so need to hook in an intermediate step to hook it).
def sample_hook(p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
    return sample_animation(p, x, conditioning, unconditional_conditioning, steps, is_img2img=False, image_conditioning=image_conditioning)

def sample_img2img_hook(p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
    return sample_animation(p, x, conditioning, unconditional_conditioning, steps, is_img2img=True, noise=noise, image_conditioning=image_conditioning)

class Script(scripts.Script):
    def title(self):
        return "Animation"

    def ui(self, is_img2img):
        prompt_end = gr.Textbox(label="End prompt", lines=1)
        num_animation_frames = gr.Slider(label="Number of frames", minimum=0, maximum=256, step=1, value=8)
        animation_levels = gr.Textbox(label="Animation levels", lines=1)
        animation_steps =  gr.Textbox(label="Animation steps", lines=1)
        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=2)
        reverse_animation = gr.Checkbox(label="Reverse animation", value=False)
        ping_pong_loop = gr.Checkbox(label="Ping-pong loop the animation back and forth", value=False)
        dont_average_animation = gr.Checkbox(label="Don't average latents", value=False)
        if is_img2img:
            img2img_img_is_first_frame = [gr.Checkbox(label="img2img image is the first frame", value=False)] # There's a bug where this generates noise that I'm still debugging.
        else:
            img2img_img_is_first_frame = []
        return [prompt_end, num_animation_frames, animation_levels, animation_steps, fps, reverse_animation, ping_pong_loop, dont_average_animation] + img2img_img_is_first_frame
    
    def run(self, p, prompt_end, num_animation_frames, animation_levels, animation_steps, fps, reverse_animation, ping_pong_loop, dont_average_animation, img2img_img_is_first_frame = False):
        animation_info = {
        "prompt_end":prompt_end,
        "num_animation_frames":num_animation_frames,
        "animation_levels":animation_levels,
        "animation_steps":animation_steps,
        "fps":fps,
        "reverse_animation": reverse_animation,
        "ping_pong_loop": ping_pong_loop,
        "img2img_img_is_first_frame":img2img_img_is_first_frame,
        "dont_average_animation":dont_average_animation
        }
        p.do_not_save_grid = True # Don't save a huge grid.
        p.decode_batch_one_by_one = True # This is required because no GPU can ever process all frames in a single batch. (when there are a lot of frames)
        p.img2img_img_is_first_frame = img2img_img_is_first_frame
        p.dont_average_animation = dont_average_animation
        
        if "enable_hr" in dir(p) and p.enable_hr:
            raise NotImplementedError("Animation didn't implement high resolution fix yet")
            
        if p.n_iter != 1:
            raise Exception("Animation assumes batch_count == 1", p.n_iter)
            
        modules.processing.fix_seed(p)
        
        p.prompt_end = prompt_end
        p.num_animation_frames = num_animation_frames
        p.animation_levels = animation_levels
        p.animation_steps = animation_steps
        
        steps = p.steps
        if type(p) is StableDiffusionProcessingImg2Img:
            fake_steps, t_enc = modules.sd_samplers.setup_img2img_steps(p, None)
            steps = t_enc + 1
            self.real_steps = steps
            p.fake_steps = fake_steps
            p.t_enc = t_enc
            
        # Initialize animation levels:
        if p.animation_levels:
            p.levels_sizes = [int(x) for x in p.animation_levels.split(",")]
            if p.levels_sizes[0] != 1:
                p.levels_sizes = [1] + p.levels_sizes
            if p.levels_sizes[-1] < num_animation_frames:
                p.levels_sizes.append(num_animation_frames)
            print(f"custom levels: {p.levels_sizes}")
        else:
            max_tree_height = int_log2(num_animation_frames)
            if num_animation_frames != (1<<max_tree_height):
                max_tree_height+=1
                num_animation_frames = (1<<max_tree_height)
                print(f"changing num animation frames to a power of 2: {num_animation_frames}")
                
            p.levels_sizes = [(1 << i) for i in range(max_tree_height + 1)] # TODO: want those as parameters?
            p.animation_levels = ",".join(str(a) for a in p.levels_sizes)
            
        num_levels = len(p.levels_sizes)
        
        # Initialize steps per level:
        if num_levels == 1: # For single picture
            steps_per_level = [0]
            p.sigmas_per_level = [0]
        elif p.animation_steps:
            steps_per_level = [int(x) for x in p.animation_steps.split(",")]
            p.sigmas_per_level=[0]
            s = 0
            for i in range(len(steps_per_level)):
                s += steps_per_level[i]
                p.sigmas_per_level.append(s)
            if s > steps:
                raise Exception("Bad animation steps - too many steps!", s, p.animation_steps, p.sigmas_per_level)
            if len(p.sigmas_per_level) != num_levels:
                raise Exception("Bad animation_steps", num_levels, len(p.sigmas_per_level), p.animation_steps, p.sigmas_per_level)
        else:
            p.sigmas_per_level = list(range(0, steps, steps // num_levels))
            steps_per_level = [p.sigmas_per_level[i] - p.sigmas_per_level[i-1] for i in range(1, num_levels)]
        p.sigmas_per_level.append(steps)
        
        total_steps = 0
        for i in range(num_levels):
            total_steps += (p.sigmas_per_level[i+1] - p.sigmas_per_level[i]) * p.levels_sizes[i]
        
        # update the progress bars:
        shared.total_tqdm.updateTotal(total_steps)
        shared.total_tqdm.use_for_ui_progress_bar = True
        
        print(f"steps_per_level: {steps_per_level}")
        
        p.sample_hook = sample_hook
        p.sample_img2img_hook = sample_img2img_hook
        processed = process_images(p)
        
        _, output_video_path = images.save_image(None, opts.outdir_videos, "", p.seed, p.prompt + " - " + prompt_end, "", p=p, return_path_only=True)
        output_video_path = os.path.realpath(output_video_path + VIDEO_EXTENSION)
        with open(output_video_path + ".txt","w") as animation_params_file:
            json.dump(animation_info, animation_params_file)
            
        frames = processed.images
        if reverse_animation:
            frames = frames[::-1]
            
        if ping_pong_loop:
            frames += frames[::-1]
            
        save_animation(frames, output_video_path, fps=fps)
        processed.video =  output_video_path
        return processed
