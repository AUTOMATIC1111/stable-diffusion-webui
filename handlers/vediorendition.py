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
from filestorage.__init__ import push_local_path
import datetime
import os
import random

class VideoRenditionTaskType(Txt2ImgTask):
    Rendition = 1  # 视频风格转换

class VideoRenditionTaskHandler(Txt2ImgTaskHandler):
    def __init__(self):
        super(VideoRenditionTaskHandler, self).__init__()
        self.task_type = TaskType.VideoRendition

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        # 根据任务的不同类型：执行不同的任务
        if task.minor_type == VideoRenditionTaskType.Rendition:
            # yield from self._exec_rendition(task)
            yield from self._exec_rendition(task)
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
            video_rendition(base_model_path,input_video_path,output_video_path)
            progress.status = TaskStatus.Uploading
            current_date = datetime.datetime.now()
            formatted_date = current_date.strftime('%Y-%m-%d')
            file_name= os.path.basename(output_video_path)
            remoting,local=f'media/{formatted_date}/{file_name}',output_video_path
            oss_key=push_local_path(remoting,local)
            yield progress

            progress = TaskProgress.new_finish(task, {
                    'rednition_video_key': oss_key
                })
            progress.task_desc = f'VideoRendition task:{task.id} finished.'
            yield progress

        except  Exception as e:
            progress.status = TaskStatus.Failed
            progress.task_desc = f'VideoRendition task:{task.id} failed.{e}'
            yield progress

