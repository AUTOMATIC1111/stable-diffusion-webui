import os
import sys
import modules.scripts as scripts
import gradio as gr
import math
import random
import re
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    def title(self):
        return "Seed travel"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        seed_travel_extra = []

        dest_seed = gr.Textbox(label='Destination seed(s) (Comma separated)', lines=1)
        rnd_seed = gr.Checkbox(label='Only use Random seeds (Unless comparing paths)', value=False)
        seed_count = gr.Number(label='Number of random seed(s)', value=4)
        compare_paths = gr.Checkbox(label='Compare paths (Separate travels from 1st seed to each destination)', value=False)
        steps = gr.Number(label='Steps', value=10)
        loopback = gr.Checkbox(label='Loop back to initial seed', value=False)
        save_video = gr.Checkbox(label='Save results as video', value=True)
        video_fps = gr.Number(label='Frames per second', value=30)
        bump_seed = gr.Slider(label='Bump seed (If > 0 do a Compare Paths but only one image. No video)', value=0.0, minimum=0, maximum=1, step=0.01)
        show_images = gr.Checkbox(label='Show generated images in ui', value=True)
        unsinify = gr.Checkbox(label='"Hug the middle" during interpolation', value=False)
        allowdefsampler = gr.Checkbox(label='Allow the default Euler a Sampling method. (Does not produce good results)', value=False)

        return [rnd_seed, seed_count, dest_seed, steps, unsinify, loopback, save_video, video_fps, show_images, compare_paths, allowdefsampler, bump_seed]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def run(self, p, rnd_seed, seed_count, dest_seed, steps, unsinify, loopback, save_video, video_fps, show_images, compare_paths, allowdefsampler, bump_seed):
        initial_info = None
        images = []

        # If we are just bumping seeds, ignore compare_paths and save_video
        if bump_seed > 0:
            compare_paths = False
            save_video = False
            steps = 1
            allowdefsampler = True # Since we aren't trying to get to a target seed, this will be ok.

        if not allowdefsampler and p.sampler_index == 0:
            print(f"You seem to be using Euler a, it will not produce good results.")
            return Processed(p, images, p.seed)

        if rnd_seed and (not seed_count or int(seed_count) < 2):
            print(f"You need at least 2 random seeds.")
            return Processed(p, images, p.seed)

        if not rnd_seed and not dest_seed:
            print(f"No destination seeds were set.")
            return Processed(p, images, p.seed)

        if not save_video and not show_images:
            print(f"Nothing to show in gui. You will find the result in the ouyput folder.")
            #return Processed(p, images, p.seed)

        if save_video:
            import numpy as np
            try:
                import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
            except ImportError:
                print(f"moviepy python module not installed. Will not be able to generate video.")
                return Processed(p, images, p.seed)

        # Remove seeds within () to help testing
        dest_seed = re.sub('\([^)]*\)', ',', dest_seed)
        dest_seed = re.sub(',,*', ',', dest_seed)

        # Custom seed travel saving
        travel_path = os.path.join(p.outpath_samples, "travels")
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        travel_path = os.path.join(travel_path, f"{travel_number:05}")
        p.outpath_samples = travel_path

        # Force Batch Count and Batch Size to 1.
        p.n_iter = 1
        p.batch_size = 1

        if compare_paths or bump_seed > 0:
            loopback = False

        # Random seeds
        if rnd_seed == True:
            seeds = []          
            if compare_paths and not p.seed == None:
                seeds.append(p.seed)
            s = 0          
            while (s < seed_count):
                seeds.append(random.randint(0,2147483647))
                #print(seeds)
                s = s + 1
        # Manual seeds        
        else:
            seeds = [] if p.seed == None else [p.seed]
            seeds = seeds + [int(x.strip()) for x in dest_seed.split(",")]
        if loopback:
            seeds.append(seeds[0])
        p.seed = seeds[0]
        
        total_images = (int(steps) * len(seeds)) - (0 if loopback else (int(steps) - 1))
        print(f"Generating {total_images} images.")

        # Set generation helpers
        state.job_count = total_images

        for s in range(len(seeds)-1):
            if state.interrupted:
                break
            if not (compare_paths or bump_seed):
                p.seed = seeds[s]
            p.subseed = seeds[s+1] if s+1 < len(seeds) else seeds[0]
            fix_seed(p)
            # We want to save seeds since they might have been altered by fix_seed()
            seeds[s] = p.seed
            if s+1 < len(seeds): seeds[s+1] = p.subseed

            numsteps = 1 if not loopback and s+1 == len(seeds) else int(steps) # Number of steps is 1 if we aren't looping at the last seed
            if compare_paths and numsteps == 1:
                numsteps = 0
            step_images = []
            for i in range(numsteps):
                if state.interrupted:
                    break
                if bump_seed > 0:
                    p.subseed_strength = bump_seed
                elif unsinify:
                    x = float(i/float(steps))
                    p.subseed_strength = x + (0.1 * math.sin(x*2*math.pi))
                else:
                    p.subseed_strength = float(i/float(steps))
                proc = process_images(p)
                if initial_info is None:
                    initial_info = proc.info
                step_images += proc.images
                images += proc.images

            if save_video and compare_paths and numsteps > 1:
                clip = ImageSequenceClip.ImageSequenceClip([np.asarray(t) for t in step_images], fps=video_fps)
                clip.write_videofile(os.path.join(travel_path, f"travel-{travel_number:05}-{s:04}.mp4"), verbose=False, logger=None)

        if save_video and not compare_paths:
            clip = ImageSequenceClip.ImageSequenceClip([np.asarray(t) for t in images], fps=video_fps)
            clip.write_videofile(os.path.join(travel_path, f"travel-{travel_number:05}.mp4"), verbose=False, logger=None)

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        return processed

    def describe(self):
        return "Travel between two (or more) seeds and create a picture at each step."
