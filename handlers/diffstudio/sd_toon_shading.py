
from .diffsynth.models import ModelManager
from .diffsynth.pipelines.stable_diffusion_video import SDVideoPipeline
from .diffsynth.controlnets.controlnet_unit import  ControlNetConfigUnit
from .diffsynth.data.video import  VideoData,save_video
from .diffsynth.extensions.RIFE import RIFESmoother
import torch
from PIL import Image
import numpy as np
import os
import random
import string
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import datetime
from loguru import logger
from datetime import date
import time
import cv2
import tempfile
import math
from modules import shared, scripts
model_dir=os.path.join(scripts.basedir(), "models")

# Load models
def rendition_video(base_model_path,input_video_path,width=0,height=0,fps=0):
    logger.info(f'video rendition begin.....')
    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
    model_manager.load_textual_inversions(model_dir+ os.path.sep +"videorendition/textual_inversion")
    model_manager.load_models([
        base_model_path,
        model_dir+ os.path.sep +"animatediff/mm_sd_v15_v2.ckpt",
        model_dir+ os.path.sep +"ControlNet/control_v11p_sd15_lineart.pth",
        model_dir+ os.path.sep +"ControlNet/control_v11f1e_sd15_tile.pth",
        model_dir+ os.path.sep +"videorendition/RIFE/flownet.pkl"
    ])
    pipe = SDVideoPipeline.from_model_manager(
        model_manager,
        [
            ControlNetConfigUnit(
                processor_id="lineart",
                model_path=model_dir+ os.path.sep +"ControlNet/control_v11p_sd15_lineart.pth",
                scale=0.5
            ),
            ControlNetConfigUnit(
                processor_id="tile",
                model_path=model_dir+ os.path.sep +"ControlNet/control_v11f1e_sd15_tile.pth",
                scale=0.5
            )
        ]
    )
    smoother = RIFESmoother.from_model_manager(model_manager)
    # TODO 视频处理：width，height，fps
    if width==0 or height==0 or fps==0:
        cap = cv2.VideoCapture(input_video_path)
        width= cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if fps==0:
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

    video = VideoData(
        video_file=input_video_path,
        height=height, width=width)
    
    index=0
    output_videos=[]
    temp_dir = tempfile.mkdtemp()
    for sta in range(0,len(video),300):
        logger.info(f'rendition ing....,{index+1}/{math.ceil(len(video)/300)}')
        index+=1
        if sta+300>len(video):
            input_video = [video[i] for i in range(sta, len(video))]
        else:
            input_video = [video[i] for i in range(sta, sta+300)]
        # Toon shading (20G VRAM)
        torch.manual_seed(0)
        output_video_path_split=temp_dir+"/"+"_"+str(index)+".mp4"
        output_video = pipe(
            prompt="best quality, perfect anime illustration,light,",
            negative_prompt="verybadimagenegative_v1.3",
            cfg_scale=3, clip_skip=2,
            controlnet_frames=input_video, num_frames=len(input_video),
            num_inference_steps=7, height=height, width=width,
            animatediff_batch_size=32, animatediff_stride=16,
            vram_limit_level=0,
        )
        output_video = smoother(output_video)
        # Save video
        save_video(output_video, output_video_path_split, fps=fps)
        output_videos.append(output_video_path_split)
    
    del model_manager,smoother,video

    return output_videos


def name_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

# 合并音频
def video_music(video1_path, video2_path2, save_path):
    logger.info(f'Add music ing.....')
    video1, video2 = VideoFileClip(video1_path), VideoFileClip(video2_path2)
    audio1 = video1.audio  # 提取第一个视频的音频部分(可截取部分音频，使用subclip()先将读取的视频文件截取后再操作即可)
    # audio1.write_audiofile('audio.mp3') # 可以保存提取的音频为MP3格式文件
    video3 = video2.set_audio(audio1)  # 将提取的视频1的音频合成到2视频中
    video3.write_videofile(save_path)

# 合并视频
def merge_video(video_files,output_filename):
    logger.info(f'Merge video slices.....')
    clips = [VideoFileClip(file) for file in video_files]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_filename)

def size_control(input_video_path):
    # 支持64的整数倍
    cap = cv2.VideoCapture(input_video_path)
    width= cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height= cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width=int((width//64)*64)
    height=int((height//64)*64)
    return width,height,fps

    
def video_rendition(base_model_path,input_video_path,output_video_path):
    current_time1 = datetime.datetime.now()
    file_name, file_ext = os.path.splitext(os.path.basename(base_model_path))
    temp_dir = tempfile.mkdtemp()
    output_filename=temp_dir+"/"+file_name+"_merge.mp4"
    output_filename_music=output_video_path
    width,height,fps=size_control(input_video_path)
    output_videos=rendition_video(base_model_path,input_video_path,width=width,height=height,fps=fps)
    merge_video(output_videos,output_filename)
    video_music(input_video_path, output_filename, output_filename_music)
    current_time2 = datetime.datetime.now()
    time_diff=current_time2-current_time1
    minutes_diff =time_diff.total_seconds() / 60
    logger.info(f'video rendition finlished:{input_video_path},it took : {minutes_diff} minutes.')
    return output_filename_music

if __name__ == '__main__':
    base_model_path=r'/root/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors'
    input_video_path=r'/root/stable-diffusion-webui/handlers/diffstudio/1.mp4'
    output_filename_path=video_rendition(base_model_path,input_video_path)
    print(output_filename_path)
