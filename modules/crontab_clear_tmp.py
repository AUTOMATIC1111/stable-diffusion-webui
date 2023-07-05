import os, time, shutil
from datetime import datetime

from threading import Timer
from pathlib import Path
import tempfile


def __delete_files(path, expiry_timestamp=3600):
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
                    print(
                        f"文件已过期：now={time.strftime('%Y-%m-%d %H:%M:%S', now())},creation_timestamp={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_timestamp))},diff_seconds={diff_seconds},file_name={file}")
                else:
                    print(
                        f"文件未过期：creation_timestamp={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_timestamp))},diff_seconds={diff_seconds},file_name={file}")
            for dir in dirs:
                if len(os.listdir(os.path.join(path, dir))) == 0:
                    shutil.rmtree(os.path.join(path, dir))
        return True
    except Exception as e:
        print(e)
        return False


def clear_gradio_tmp():
    path = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")
    __delete_files(path)
    t = Timer(3, clear_gradio_tmp)
    t.start()

