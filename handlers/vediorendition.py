import typing
from loguru import logger
from PIL import ImageOps, Image
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed, fix_seed
from handlers.utils import init_script_args, get_selectable_script, init_default_script_args, \
    load_sd_model_weights, save_processed_images, get_tmp_local_path, get_model_local_path
import tempfile
from handlers.diffstudio.sd_toon_shading import video_rendition
from handlers.roop.ins_swap_face import exec_roop_video
from filestorage.__init__ import push_local_path
import datetime
import os
import random

class VideoRenditionTaskType(Txt2ImgTask):
    Rendition = 1  # 视频风格转换
    SwapFace=2

class VideoRenditionTaskHandler(Txt2ImgTaskHandler):
    def __init__(self):
        super(VideoRenditionTaskHandler, self).__init__()
        self.task_type = TaskType.VideoRendition

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        # 根据任务的不同类型：执行不同的任务
        if task.minor_type == VideoRenditionTaskType.Rendition:
            # yield from self._exec_rendition(task)
            yield from self._exec_rendition(task)
        if task.minor_type == VideoRenditionTaskType.SwapFace:
            yield from self._exec_swapface(task)
    def _exec_rendition(self, task: Task) -> typing.Iterable[TaskProgress]:

        logger.info(f"VideoRendition beigin.....,{task['task_id']}")
        logger.info(f"VideoRendition get_local_checkpoint.....,{task['task_id']}")
        base_model_path = self._get_local_checkpoint(task)
        path=task['video']
        logger.info(f"VideoRendition get_local_video.....,{path}")
        input_video_path = get_tmp_local_path(path)
        progress = TaskProgress.new_ready(
            task, f'model loaded, run video rendition...')
        yield progress

        random_number = random.randint(10000, 99999)
        random_name =str(random_number)
        output_video_path = tempfile.NamedTemporaryFile(prefix=random_name,suffix='.mp4').name

        progress.status = TaskStatus.Running
        progress.task_desc = f'VideoRendition task({task.id}) running'
        yield progress

        try:
            output_filename_music=video_rendition(base_model_path,input_video_path,output_video_path)
            progress.status = TaskStatus.Uploading
            current_date = datetime.datetime.now()
            formatted_date = current_date.strftime('%Y-%m-%d')
            file_name= os.path.basename(output_filename_music)
            remoting,local=f'media/{formatted_date}/{file_name}',output_filename_music
            oss_key=push_local_path(remoting,local)
            logger.info(f"VideoRendition push_local_path.....,{oss_key}")
            yield progress

            progress = TaskProgress.new_finish(task, {
                    'video_key': remoting
                })
            progress.task_desc = f'VideoRendition task:{task.id} finished.'
            yield progress

        except  Exception as e:
            progress.status = TaskStatus.Failed
            progress.task_desc = f'VideoRendition task:{task.id} failed.{e}'
            yield progress

    def _exec_swapface(self, task: Task) -> typing.Iterable[TaskProgress]:
        
        #输入：视频，换脸目标图片，视频中要换的引用图片
        logger.info(f"VideoSwapFace beigin.....,{task['task_id']}")
        progress = TaskProgress.new_ready(
            task, f'get_local_file finished, run video swap face ...')
        yield progress

        target_path=task['target_img']
        reference_path=task['reference_img']
        video_path=task['video']
        restore=False if 'restore' not in task else task['restore']
        facial_strength=0.8 if 'facial_strength' not in task else task['facial_strength']

        logger.info(f"VideoSwapFace get_local_file.....,{target_path,reference_path,video_path}")
        target_face = get_tmp_local_path(target_path)
        reference_face = get_tmp_local_path(reference_path)
        input_video = get_tmp_local_path(video_path)
        progress.status = TaskStatus.Running
        progress.task_desc = f'VideoSwapFace task({task.id}) running'
        try:
            output_video_path=exec_roop_video(target_face,reference_face,input_video,restore=restore,facial_strength=facial_strength)
            progress.status = TaskStatus.Uploading
            current_date = datetime.datetime.now()
            formatted_date = current_date.strftime('%Y-%m-%d')
            file_name= os.path.basename(output_video_path)
            remoting,local=f'media/{formatted_date}/{file_name}',output_video_path
            oss_key=push_local_path(remoting,local)
            yield progress

            progress = TaskProgress.new_finish(task, {
                    'video_key': oss_key
                })
            progress.task_desc = f'VideoSwapFace task:{task.id} finished.'
            yield progress
        except Exception as e:
            progress.status = TaskStatus.Failed
            progress.task_desc = f'VideoSwapFace task:{task.id} failed.{e}'
            yield progress
