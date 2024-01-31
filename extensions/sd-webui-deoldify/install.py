'''
Author: SpenserCai
Date: 2023-07-28 14:37:09
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-04 09:47:33
Description: file content
'''
import os
import launch
from modules import paths_internal
import urllib.request
from tqdm import tqdm
# 从huggingface下载权重

models_dir = os.path.join(paths_internal.models_path, "deoldify")
stable_model_url = "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth"
artistic_model_url = "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeArtistic_gen.pth"
video_model_url = "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth"
stable_model_name = os.path.basename(stable_model_url)
artistic_model_name = os.path.basename(artistic_model_url)
video_model_name = os.path.basename(video_model_url)
stable_model_path = os.path.join(models_dir, stable_model_name)
artistic_model_path = os.path.join(models_dir, artistic_model_name)
video_model_path = os.path.join(models_dir, video_model_name)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(stable_model_path):
    download(stable_model_url, stable_model_path)


if not os.path.exists(artistic_model_path):
    download(artistic_model_url, artistic_model_path)

if not os.path.exists(video_model_path):
    download(video_model_url, video_model_path)

for dep in ['wandb','fastai==1.0.60', 'tensorboardX', 'ffmpeg', 'ffmpeg-python', 'yt-dlp', 'opencv-python','Pillow']:
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for DeOldify extension")
