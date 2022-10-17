from modules import devices
from modules import modelloader
from modules import shared
import threading
def save_pts(filename):
    queue_lock = threading.Lock()
    def wrap_gradio_gpu_call(func, extra_outputs=None):
        def f(*args, **kwargs):
            devices.torch_gc()
            shared.state.sampling_step = 0
            shared.state.job_count = -1
            shared.state.job_no = 0
            shared.state.job_timestamp = shared.state.get_job_timestamp()
            shared.state.current_latent = None
            shared.state.current_image = None
            shared.state.current_image_sampling_step = 0
            shared.state.skipped = False
            shared.state.interrupted = False
            shared.state.textinfo = None
            with queue_lock:
                res = func(*args, **kwargs)
            shared.state.job = ""
            shared.state.job_count = 0
            devices.torch_gc()
            #return res
        #return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)
    def copy(filename):
        import shutil,os
        try:
            os.makedirs("/content/drive/MyDrive/StableDiffusionTraining/{}/{}".format(
                filename.split("/")[-1].split(".")[0].split("-")[0],
                filename.split("/")[-1].split(".")[0],exist_ok=True
            ))
        except:
            pass
        try:
            shutil.copy(filename, "/content/drive/MyDrive/StableDiffusionTraining/{}/{}".format(
                filename.split("/")[-1].split(".")[0].split("-")[0],
                filename.split("/")[-1].split(".")[0],exist_ok=True))
            print("\n保存文件至Google Drive: "+filename)
        except Exception as e:
            import traceback
            print("\n保存训练模型至Google Drive时出错：{}".format(e))
    wrap_gradio_gpu_call(copy(filename), extra_outputs=None)

def save_outcsv(filename):
    queue_lock = threading.Lock()
    def wrap_gradio_gpu_call(func, extra_outputs=None):
        def f(*args, **kwargs):
            devices.torch_gc()
            shared.state.sampling_step = 0
            shared.state.job_count = -1
            shared.state.job_no = 0
            shared.state.job_timestamp = shared.state.get_job_timestamp()
            shared.state.current_latent = None
            shared.state.current_image = None
            shared.state.current_image_sampling_step = 0
            shared.state.skipped = False
            shared.state.interrupted = False
            shared.state.textinfo = None
            with queue_lock:
                res = func(*args, **kwargs)
            shared.state.job = ""
            shared.state.job_count = 0
            devices.torch_gc()
    def copy_csv(filename):
        import pandas,os,shutil
        model_name = filename.split("/")[2]
        short_filename = filename.split("/")[-1]
        logdir = "/content/drive/MyDrive/StableDiffusionTraining/log/{}/{}"
        try:
            os.makedirs("/content/drive/MyDrive/StableDiffusionTraining/log/{}".format(model_name))
        except:
            pass
        new_data = pandas.read_csv(filename)
        if not os.path.exists(logdir.format(model_name, short_filename)):
            shutil.copy(filename, logdir.format(model_name, short_filename))
            print("\n已经保存loss的log文件")
        else:
            old_data=pandas.read_csv(logdir.format(model_name, short_filename))
            frames=[new_data, old_data]
            merged = pandas.concat(frames, ignore_index=True)
            os.remove(logdir.format(model_name, short_filename))
            merged.to_csv(logdir.format(model_name, short_filename))
            print("\n已经合并loss的log文件")

