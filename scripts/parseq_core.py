from distutils.log import info
import string
import numpy as np
from tqdm import trange
import glob
import os
import subprocess
from PIL import Image, ImageDraw
import logging
import ffmpeg
import json
import cv2
from math import pi
import os
import sys
from subprocess import Popen, PIPE
from skimage import exposure
import re
import math
from datetime import datetime
import textwrap

class Parseq():

    def run(self, p, input_img, input_path:string, output_path:string, save_images:bool, dry_run_mode:bool,
            overlay_metadata:bool, default_output_dir:string, param_script_string:string, sd_processor):
        # TODO - batch count & size support (only useful is seed is random)

        output_path = get_output_path(output_path, default_output_dir)
        logging.info(f"Output will be written to: {output_path}")

        input_type = 'video'
        if (not input_path):
            logging.info("No input video supplied, assuming loopback.")
            input_type = 'loopback'
            if (not input_img):
                error_message = "You must supply a path to an input video or an input image for loopback."
                logging.error(error_message)
                return [None, error_message]

        if (not param_script_string):
            error_message = "You must supply JSON paramter script."
            return [None, error_message]

        # Load param_script
        param_script, options = load_param_script(param_script_string)
        logging.info(options)
       
        # Get input frame info
        input_fps = None
        if (input_type == 'video'):
            input_frames = video_frames(input_path)
            input_width = video_width(input_path)
            input_height = video_height(input_path)
            source_fps = video_fps(input_path)
            input_fps =  parseIntOrDefault(options['input_fps'], source_fps)
            logging.info(f"Input: {input_frames} frames ({input_width}x{input_height} @ {input_fps}fps (source: {source_fps}fps))")
            output_fps = parseIntOrDefault(options['output_fps'], (input_fps or 20))
        else:
            # Loopback
            input_width = p.width
            input_height = p.height
            output_fps = parseIntOrDefault(options['output_fps'], 20)
            input_fps =  parseIntOrDefault(options['input_fps'], output_fps)

        cc_window_width = parseIntOrDefault(options['cc_window_width'], 0)
        cc_window_rate = parseFloatOrDefault(options['cc_window_slide_rate'], 1)
        cc_include_initial_image = bool(options['cc_use_input'])
        logging.info(f"Loaded options: input_type:{input_type}; input_fps:{input_fps}; output_fps:{output_fps}; cc_window_width:{cc_window_width}; cc_window_rate:{cc_window_rate}; cc_include_initial_image:{cc_include_initial_image}")


        # TODO: Compare input frame count to scripted frame count and decide what to do if they don't match.
        #       For now we just stop on shortest.
        # param_script_frames= max(param_script.keys())
        # logging.info(f"Script frames: {param_script_frames}, input frames: {input_frames}")
        # frame_ratio = param_script_frames / float(input_frames)
        # if frame_ratio < 1:
        #     logging.warning(f"Some input frames will be skipped to match script frame count. Ratio: {frame_ratio}")
        # elif frame_ratio > 1:
        #     logging.warning(f"Some input frames will be duplicated to match script frame count. Ratio: {frame_ratio}")

        # Init video in
        if (input_type == 'video'):
            process1 = (
                ffmpeg
                .input(input_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=input_fps)
                .run_async(pipe_stdout=True)
            )

        # Init video out
        process2 = (
            ffmpeg
            .input('pipe:', framerate=input_fps, format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(p.width, p.height))
            .output(output_path, pix_fmt='yuv420p', r=output_fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        frame_pos = 0
        out_frame_history=[]
        while True:
            if not frame_pos in param_script:
                logging.info(f"Ending: no script information about how to process frame {frame_pos}.")
                if (input_type == 'video'):
                    process1.stdin.close()
                break     

            # Read frame
            if (input_type == 'video'):
                in_bytes = process1.stdout.read(input_width * input_height * 3)
                if not in_bytes:
                    logging.info(f"Ending: no further video input at frame {frame_pos}.")
                    
                    break            
                in_frame = (
                    np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([input_height, input_width, 3])
                )
                if frame_pos == 0:
                    initial_input_image = in_frame.copy()
            else: # loopback                  
                if frame_pos == 0:
                    in_frame = np.asarray(input_img)
                    initial_input_image = in_frame
                else:
                    in_frame = out_frame_history[frame_pos-1]

            # Resize
            in_frame_resized = cv2.resize(in_frame, (p.width, p.height), interpolation = cv2.INTER_LANCZOS4)

            # Blend historical frames
            start_frame_pos = round(clamp(0, frame_pos-param_script[frame_pos]['loopback_frames'], len(out_frame_history)-1))
            end_frame_pos = round(clamp(0, frame_pos-1, len(out_frame_history)-1))
            frames_to_blend = [in_frame_resized] + out_frame_history[start_frame_pos:end_frame_pos]
            blend_decay = clamp(0, param_script[frame_pos]['loopback_decay'], 1)
            logging.debug(f"Blending {len(frames_to_blend)} frames (current frame plus {start_frame_pos} to {end_frame_pos}) with decay {blend_decay}.")
            in_frame_blended = self.blend_frames(frames_to_blend, blend_decay)

            #Rotate, zoom & pan (x,y,z)
            in_frame_rotated = ImageTransformer().rotate_along_axis(in_frame_blended, param_script[frame_pos]['rotx'], param_script[frame_pos]['roty'], param_script[frame_pos]['rotz'],
            -param_script[frame_pos]['panx'], -param_script[frame_pos]['pany'], -param_script[frame_pos]['zoom'])
        
            # Color correction
            cc_window_start, cc_window_end  = self.compute_cc_target_window(frame_pos, cc_window_width, cc_window_rate)
            if (cc_window_end>0):
                cc_target_images = out_frame_history[cc_window_start:cc_window_end]
            else:
                cc_target_images = []
            if (cc_include_initial_image):
                cc_target_images.append(initial_input_image)

            cc_target_histogram = compute_cc_target(cc_target_images)
            if cc_target_histogram is None:
                logging.debug(f"Skipping color correction on frame {frame_pos} (target frames: {cc_window_start} to {cc_window_end})")
                in_frame_cc = in_frame_rotated
            else:
                logging.debug(f"Applying color correction on frame {frame_pos} (target frames: {cc_window_start} to {cc_window_end}) effective window size: {len(cc_target_images)})")
                in_frame_cc = apply_color_correction(in_frame_rotated, cc_target_histogram)
            
            final_in_frame = in_frame_cc
            # Do SD 
            # TODO - batch count & batch size support: for each batch, for each batch_item          
            p.n_iter = 1
            p.batch_size = 1
            p.init_images = [Image.fromarray(final_in_frame)] 
            p.seed = math.floor(param_script[frame_pos]['seed'])
            p.subseed = param_script[frame_pos]['subseed']
            p.subseed_strength = param_script[frame_pos]['subseed_strength']
            p.scale = clamp(-100, param_script[frame_pos]['scale'], 100)
            p.denoising_strength = clamp(0.01, param_script[frame_pos]['denoise'], 1)
            p.prompt = param_script[frame_pos]['positive_prompt'] 
            p.negative_prompt = param_script[frame_pos]['negative_prompt'] 
            #for name, value in param_script[frame_pos]:
            logging.info(param_script[frame_pos])
            p.extra_generation_params=param_script[frame_pos]
            p.extra_generation_params['input_type']=input_type

            logging.info(f"[{frame_pos}] - seed:{p.seed}; subseed:{p.subseed}; subseed_strength:{p.subseed_strength}; scale:{p.scale}; ds:{p.denoising_strength}; prompt: {p.prompt}; negative_prompt: {p.negative_prompt}")
            if dry_run_mode:
                processed_image = Image.fromarray(final_in_frame)
            else:
                processed = sd_processor.process_images(p)
                processed_image = processed.images[0]
            
            #overlay metadata
            out_frame_with_metadata = None
            if (overlay_metadata):
                processed_image_with_metadata = processed_image.copy()
                draw = ImageDraw.Draw(processed_image_with_metadata)
                draw.text((10, 10), textwrap.fill(json.dumps(p.extra_generation_params, indent=2),64))
                frame_to_render = np.asarray(processed_image_with_metadata)
                frame_to_loop_back = np.asarray(processed_image)
            else:
                frame_to_render = np.asarray(processed_image)
                frame_to_loop_back = frame_to_render
            
            #Save frame
            process2.stdin.write(
                frame_to_render
                .astype(np.uint8)
                .tobytes()
            )
            if (save_images):
                cv2.imwrite(f"{output_path}-{frame_pos:05}.png", cv2.cvtColor(frame_to_render, cv2.COLOR_RGB2BGR))

            # Save frames for loopback
            out_frame_history.append(frame_to_loop_back)
            frame_pos += 1

        logging.info("About to close video streams.")
        process2.stdin.close()
        if (input_type == 'video'):
            process1.wait()
        process2.wait()

        return [out_frame_history, "Here's some info mate. Where you gonna put it mate?"]


    def compute_cc_target_window(self, current_pos, window_size, window_rate):
        cc_window_end = round((current_pos)*window_rate)
        if window_size == -1:
            cc_window_start = 0
        else:
            cc_window_start = max(0, cc_window_end-window_size)
        return cc_window_start, cc_window_end


    def blend_frames(self, frames_to_blend, decay):
        if len(frames_to_blend) == 1:
            return frames_to_blend[0]
        return cv2.addWeighted(frames_to_blend[0], (1-decay), self.blend_frames(frames_to_blend[1:], decay), decay, 0)

#### Image conversion utils
def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return np.asarray(img)

def compute_cc_target(target_images):
    if target_images is None or len(target_images)==0:
        return None

    target_histogram = np.zeros(np.shape(target_images[0])).astype('float64')
    for img in target_images:
        target_histogram_component = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2LAB).astype('float64')
        target_histogram += (target_histogram_component/len(target_images)).astype('float64')
                
    target_histogram=target_histogram.astype('uint8')
    
    return target_histogram

def apply_color_correction(image, target):
    logging.debug("Applying color correction.")
    corrected = cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            image.copy(),
            cv2.COLOR_RGB2LAB
        ),
        target,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8")
    return corrected


#### Param script utils:
def load_param_script(param_script_string):
    json_obj = json.loads(param_script_string)
    rendered_frames_raw = json_obj['rendered_frames']
    param_script = dict()
    for event in rendered_frames_raw:
        if event['frame'] in param_script:
            logging.debug(f"Duplicate frame {event['frame']} detected. Latest wins.")        
        param_script[event['frame']] = event

    last_frame=max(param_script.keys())
    logging.info(f"Script contains {len(param_script)} frames, last frame is {last_frame}")
    
    for f in range(0, last_frame+1):
        if not event['frame'] in param_script:
            logging.warning(f"Script should contain contiguous frame definitions, but is missing frame {f}.")

    return param_script, json_obj['options']

#### Math utils:
def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def parseIntOrDefault(input, default):
    try:
        return int(input)
    except ValueError:
        return default

def parseFloatOrDefault(input, default):
    try:
        return float(input)
    except ValueError:
        return default

#### File utils:
def get_output_path(output_path, img2img_default_output_path):
    output_path = re.sub('<timestamp>', datetime.now().strftime("%Y%m%d-%H%M%S"), output_path)
    output_path = re.sub('<img2img_output_path>', img2img_default_output_path, output_path)
    return output_path

#### Video utils:
def video_frames(video_file):
    num_frames = get_video_info(video_file)['nb_frames']
    return num_frames

def video_width(video_file):
    return get_video_info(video_file)['width']

def video_height(video_file):
    return get_video_info(video_file)['height']

def video_fps(video_file):
    return  int(get_video_info(video_file)['r_frame_rate'].split('/')[0])

def get_video_info(video_file):
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return video_info

def save_video(video_name, path = './', files=[], fps=10, smooth=True):
    video_name = path + video_name
    txt_name = video_name + '.txt'

    # save pics path in txt
    open(txt_name, 'w').write('\n'.join(["file '" + os.path.join(path, f) + "'" for f in files]))

    subprocess.call(' '.join([
        'ffmpeg/ffmpeg -y',
        f'-r {fps}',
        '-f concat -safe 0',
        f'-i "{txt_name}"',
        '-vcodec libx264',
        '-filter:v minterpolate' if smooth else '',   # smooth between images
        '-crf 10',
        '-pix_fmt yuv420p',
        f'"{video_name}"'
    ]))
    return video_name

# from https://github.com/eborboihuc/rotate_3d/blob/master/image_transformer.py
# License: https://github.com/eborboihuc/rotate_3d/blob/master/LICENSE.md
class ImageTransformer():
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        self.image = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.num_channels = img.shape[2]
        
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = self.get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz += self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        
        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height), flags=cv2.INTER_LANCZOS4  , borderMode=cv2.BORDER_REFLECT)


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    def get_rad(self, theta, phi, gamma):
        return (self.deg_to_rad(theta),
                self.deg_to_rad(phi),
                self.deg_to_rad(gamma))

    def get_deg(self, rtheta, rphi, rgamma):
        return (self.rad_to_deg(rtheta),
                self.rad_to_deg(rphi),
                self.rad_to_deg(rgamma))

    def deg_to_rad(self, deg):
        return deg * pi / 180.0

    def rad_to_deg(self, rad):
        return rad * 180.0 / pi        
