import os, time, shutil
from datetime import datetime

from threading import Timer
from pathlib import Path
import tempfile
import logging
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(module)s | %(filename)s | line:%(lineno)d] [%(levelname)s] : %(message)s'
)


def __delete_files(path, expiry_timestamp=14400):
    print(f"清理目录：path={path},过期时间={expiry_timestamp} s")
    current_timestamp = time.time()
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                creation_timestamp = os.path.getctime(file_path)
                diff_seconds = current_timestamp - creation_timestamp
                if diff_seconds > expiry_timestamp:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        shutil.rmtree(file_path)  # 删除文件或文件夹
                    logging.info(
                        f"文件已过期：now={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())},creation_timestamp={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_timestamp))},diff_seconds={diff_seconds},file_name={file}")
            for dir in dirs:
                if len(os.listdir(os.path.join(path, dir))) == 0:
                    shutil.rmtree(os.path.join(path, dir))
    except Exception as e:
        logging.exception(e)


def clear_gradio_tmp():
    path = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    __delete_files(path)
    logging.info("startting clear gradio tmp")
    timer = BackgroundScheduler()
    timer.add_job(clear_gradio_tmp(), "interval ", seconds=3600)
    timer.start()
