
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
import time
import moviepy.editor as mp
import numpy as np
from tqdm import tqdm
import insightface
import onnxruntime
import os
import math
from PIL import Image
import argparse
from loguru import logger
import shutil
from modules import shared, scripts
from moviepy.editor import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy
import random
import tempfile

VIDEO_FRAMES_DIRECTORY = "handlers/roop/_tmp_frames"
PROCESSED_FRAMES_DIRECTORY = "handlers/roop/_tmp_frames_swap"
PROCESSED_VIDEO_DIRECTORY= "handlers/roop/_tmp_video_nomusic"

# 合并音频
def video_music(video1_path, video2_path2, save_path):
    logger.info(f'Add music ing.....')
    video1, video2 = VideoFileClip(video1_path), VideoFileClip(video2_path2)
    audio1 = video1.audio  # 提取第一个视频的音频部分(可截取部分音频，使用subclip()先将读取的视频文件截取后再操作即可)
    # audio1.write_audiofile('audio.mp3') # 可以保存提取的音频为MP3格式文件
    video3 = video2.set_audio(audio1)  # 将提取的视频1的音频合成到2视频中
    video3.write_videofile(save_path)


def video_to_images(video_file_path):
    list_files = []
    os.makedirs(VIDEO_FRAMES_DIRECTORY, exist_ok=True)
    cap = cv2.VideoCapture(video_file_path)
    frame_count = 0

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_skip_ratio = 1
    real_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip_ratio == 0:
            frame_filename = os.path.join(VIDEO_FRAMES_DIRECTORY, f"{frame_count:07d}.png")
            list_files.append(frame_filename)
            cv2.imwrite(frame_filename, frame)
            real_frame_count += 1

    cap.release()

    return list_files, original_fps

def images_to_video(images_path, fps, output_file):
    clip = mp.ImageSequenceClip(images_path, fps=fps)
    clip.write_videofile(output_file, fps=fps)

def get_images_list(directory):
    return [file for file in sorted(os.listdir(directory)) if
            file.endswith((".jpg", ".jpeg", ".png"))]

def remove_video_frames_directory():
    shutil.rmtree(VIDEO_FRAMES_DIRECTORY, ignore_errors=True)
    logger.info(f"Removed directory `{VIDEO_FRAMES_DIRECTORY}`")
def remove_processed_frames_directory():
    shutil.rmtree(PROCESSED_FRAMES_DIRECTORY, ignore_errors=True)
    logger.info(f"Removed directory `{PROCESSED_FRAMES_DIRECTORY}`")
def remove_processed_video_directory():
    shutil.rmtree(PROCESSED_VIDEO_DIRECTORY, ignore_errors=True)
    logger.info(f"Removed directory `{PROCESSED_VIDEO_DIRECTORY}`")



# 按照人脸相似度排序，返回最高的，没有返回None
def find_similar_faces(many_faces,reference_face,facial_strength):
    similar_faces = None
    max_distance=0
    for face in many_faces:
        if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
            current_face_distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))            
            if current_face_distance < facial_strength and current_face_distance>max_distance:
                similar_faces=face
    return similar_faces



def process_image(image_file_path,target_face,reference_face,face_analyser,face_swapper,restore=False,face_restorer=None,facial_strength=0.8):

    image_file_name=cv2.imread(image_file_path)

    # 获取视频帧的所有人脸
    all_faces = face_analyser.get(image_file_name) # 检测到的所有人脸

    reference_face=face_analyser.get(reference_face)[0] if len(face_analyser.get(reference_face))>0 else None
    target_face=face_analyser.get(target_face)[0] if len(face_analyser.get(target_face))>0 else None

    source_face=find_similar_faces(all_faces,reference_face,facial_strength)# 找到相似度最高的那个

    # 交换人脸
    result=image_file_name.copy()
    if target_face is not None and source_face is not None:
        result = face_swapper.get(result, source_face, target_face)

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # 人脸修复
    if restore:
        original_image = result_image.copy()
        numpy_image = np.array(result_image)
        numpy_image = face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(original_image, restored_image, 1)

    output_path = os.path.join(PROCESSED_FRAMES_DIRECTORY, f"output_{os.path.basename(image_file_path)}")
    result_image.save(output_path)

    


def exec_roop_video(target_img,reference_img,input_video,restore=False,facial_strength=0.8):
    
    total_start_time = time.time()
    # Remove temp directories from previous run
    remove_video_frames_directory()
    remove_processed_frames_directory()
    remove_processed_video_directory()

    # Start splitting
    video=input_video
    logger.info(f"Splitting video `{video}` to frames")
    timer_start = time.time()
    frames, video_fps = video_to_images(video)
    logger.info(f"Video FPS: {video_fps},Total frames: {len(frames)},Splitting done in {time.time() - timer_start}s")

    # Create the output directory if it doesn"t exist
    os.makedirs(PROCESSED_FRAMES_DIRECTORY, exist_ok=True)
    video_frames_images_names = get_images_list(VIDEO_FRAMES_DIRECTORY)
    target_face = cv2.imread(target_img)
    reference_face = cv2.imread(reference_img)
    
    # load model
    providers = onnxruntime.get_available_providers()
    analyser_models_dir = os.path.join(scripts.basedir(), "models" + os.path.sep + "roop" + os.path.sep + "buffalo_l")
    swapper_models_path = os.path.join(scripts.basedir(),
                                "models" + os.path.sep + "roop" + os.path.sep + "inswapper_128.onnx")
    # analyser_models_dir=r'/data/apksamba/sd/models/roop/buffalo_l'
    # swapper_models_path=r'/data/apksamba/sd/models/roop/inswapper_128.onnx'
    face_analyser = insightface.app.FaceAnalysis(name=analyser_models_dir, providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640,640))
    face_swapper = insightface.model_zoo.get_model(swapper_models_path, providers=providers)
    face_restorer = None
    if restore:
        for restorer in shared.face_restorers:
            if restorer.name() == 'CodeFormer':
                face_restorer = restorer
                break


    # Progress bar

    for image_file_name in tqdm(video_frames_images_names):
        image_file_path=os.path.join(VIDEO_FRAMES_DIRECTORY,image_file_name)
        process_image(image_file_path,target_face,reference_face,face_analyser,face_swapper,restore=False,face_restorer=face_restorer,facial_strength=facial_strength)

    processed_frames_images = [os.path.join(PROCESSED_FRAMES_DIRECTORY, file) for file in get_images_list(PROCESSED_FRAMES_DIRECTORY)]
    logger.info(f"Making video from {len(processed_frames_images)} images")

    # merge video
    os.makedirs(PROCESSED_VIDEO_DIRECTORY, exist_ok=True)
    output=os.path.join(PROCESSED_VIDEO_DIRECTORY,"no_music.mp4")

    images_to_video(processed_frames_images, fps=video_fps,
                    output_file=output)
    


    # add audio
    random_number = random.randint(10000, 99999)
    random_name =str(random_number)
    output_video_path = tempfile.NamedTemporaryFile(prefix=random_name,suffix='.mp4').name
    video_music(input_video, output, output_video_path)
    
    # remove file
    remove_processed_frames_directory()
    remove_video_frames_directory()
    remove_processed_video_directory()

    logger.info("Video swapping completed in %s seconds" % (time.time() - total_start_time))

    return output_video_path




if __name__ == "__main__":
    # 视频换脸
    """
    1.目标脸
    2.视频截图->返回置信度前四的四张人脸
    3.四张人脸选一张-作为更换脸
    4.找到视频里每一帧的所有脸，找到与更换脸相似度最高的脸。（侧脸考虑一下）
    5.更换脸
    6.脸部修复
    7.合并视频
    8.添加音频
    """
    target_img=r'/root/fxq/stable-diffusion-webui-master/z_fxq_img/reba.jpeg'
    reference_face=r'/root/fxq/stable-diffusion-webui-master/z_fxq_img/1.jpg'
    input_video=r'/root/fxq/stable-diffusion-webui-master/z_fxq_img/1.mp4'
    exec_roop_video(target_img,reference_face,input_video)