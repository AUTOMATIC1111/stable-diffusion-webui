import copy
import os
import shutil

import cv2
import gradio as gr
import modules.scripts as scripts

from modules import images
from modules.processing import process_images
from modules.shared import opts
from PIL import Image

import numpy as np

_BASEDIR = "/controlnet-m2m"
_BASEFILE = "animation"

def get_all_frames(video_path):
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if ret:
            frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return frame_list

def get_min_frame_num(video_list):
    min_frame_num = -1
    for video in video_list:
        if video is None:
            continue
        else:
            frame_num = len(video)
            print(frame_num)
            if min_frame_num < 0:
                min_frame_num = frame_num
            elif frame_num < min_frame_num:
                min_frame_num = frame_num
    return min_frame_num

def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:
      pass
  elif new_image.shape[2] == 3:
      new_image = new_image[:, :, ::-1]
  elif new_image.shape[2] == 4:
      new_image = new_image[:, :, [2, 1, 0, 3]]
  return new_image


def save_gif(path, image_list, name, duration):
    tmp_dir = path + "/tmp/" 
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    for i, image in enumerate(image_list):
        images.save_image(image, tmp_dir, f"output_{i}")

    os.makedirs(f"{path}{_BASEDIR}", exist_ok=True)

    image_list[0].save(f"{path}{_BASEDIR}/{name}.gif", save_all=True, append_images=image_list[1:], optimize=False, duration=duration, loop=0)
    

class Script(scripts.Script):  
    
    def title(self):
        return "controlnet m2m"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        # How the script's is displayed in the UI. See https://gradio.app/docs/#components
        # for the different UI components you can use and how to create them.
        # Most UI components can return a value, such as a boolean for a checkbox.
        # The returned values are passed to the run method as parameters.
        
        ctrls_group = ()
        max_models = opts.data.get("control_net_unit_count", 3)

        with gr.Group():
            with gr.Accordion("ControlNet-M2M", open = False):
                duration = gr.Slider(label=f"Duration", value=50.0, minimum=10.0, maximum=200.0, step=10, interactive=True, elem_id='controlnet_movie2movie_duration_slider')
                with gr.Tabs():
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}"):
                            with gr.TabItem("Movie Input"):
                                ctrls_group += (gr.Video(format='mp4', source='upload', elem_id = f"video_{i}"), )
                            with gr.TabItem("Image Input"):
                                ctrls_group += (gr.Image(source='upload', brush_radius=20, mirror_webcam=False, type='numpy', tool='sketch', elem_id=f'image_{i}'), )
                            ctrls_group += (gr.Checkbox(label=f"Save preprocessed", value=False, elem_id = f"save_pre_{i}"),)        
                
        ctrls_group += (duration,)

        return ctrls_group

    def run(self, p, *args):
        # This is where the additional processing is implemented. The parameters include
        # self, the model object "p" (a StableDiffusionProcessing class, see
        # processing.py), and the parameters returned by the ui method.
        # Custom functions can be defined here, and additional libraries can be imported 
        # to be used in processing. The return value should be a Processed object, which is
        # what is returned by the process_images method.
        
        contents_num = opts.data.get("control_net_unit_count", 3)
        arg_num = 3
        item_list = []
        video_list = []
        for input_set in  [tuple(args[:contents_num * arg_num][i:i+3]) for i in range(0, len(args[:contents_num * arg_num]), arg_num)]:
            if input_set[0] is not None:
                item_list.append([get_all_frames(input_set[0]), "video"])
                video_list.append(get_all_frames(input_set[0]))
            if input_set[1] is not None:
                item_list.append([cv2.cvtColor(pil2cv(input_set[1]["image"]), cv2.COLOR_BGRA2RGB), "image"])

        save_pre = list(args[2:contents_num * arg_num:3])
        item_num = len(item_list)
        video_num = len(video_list)
        duration, = args[contents_num * arg_num:]

        frame_num = get_min_frame_num(video_list)
        if frame_num > 0:
            output_image_list = []
            pre_output_image_list = []
            for i in range(item_num):
                pre_output_image_list.append([])

            for frame in range(frame_num):
                copy_p = copy.copy(p)
                copy_p.control_net_input_image = []
                for item in item_list:
                    if item[1] == "video":
                        copy_p.control_net_input_image.append(item[0][frame])
                    elif item[1] == "image":
                        copy_p.control_net_input_image.append(item[0])
                    else:
                        continue

                proc = process_images(copy_p)
                img = proc.images[0]
                output_image_list.append(img)

                for i in range(len(save_pre)):
                    if save_pre[i]:
                        try:
                            pre_output_image_list[i].append(proc.images[i + 1])
                        except:
                            print(f"proc.images[{i} failed")

                copy_p.close()

            # filename format is seq-seed-animation.gif seq is 5 places left filled with 0

            seq = images.get_next_sequence_number(f"{p.outpath_samples}{_BASEDIR}", "")
            filename = f"{seq:05}-{proc.seed}-{_BASEFILE}"
            save_gif(p.outpath_samples, output_image_list, filename, duration)
            proc.images = [f"{p.outpath_samples}{_BASEDIR}/{filename}.gif"]


            for i in range(len(save_pre)):
                if save_pre[i]:
                    # control files add -controlX.gif where X is the controlnet number
                    save_gif(p.outpath_samples, pre_output_image_list[i], f"{filename}-control{i}", duration)
                    proc.images.append(f"{p.outpath_samples}{_BASEDIR}/{filename}-control{i}.gif")

        else:
            proc = process_images(p)
        
        return proc
