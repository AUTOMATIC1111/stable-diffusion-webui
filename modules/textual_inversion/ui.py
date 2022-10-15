import html

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared


def save_pts(filename):
    from modules import devices
    from modules import modelloader
    import threading
    import traceback
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
            os.makedirs("/content/drive/MyDrive/StableDiffusionTraining", exist_ok=True)
            os.makedirs("/content/drive/MyDrive/StableDiffusionTraining/{}".format(
                filename.split("/")[-1].split(".")[0],exist_ok=True
            ))
        except:
            pass
        try:
            shutil.copy(filename, "/content/drive/MyDrive/StableDiffusionTraining/{}".format(
                    filename.split("/")[-1].split(".")[0]))
        except Exception as e:
            print("保存训练模型至Google Drive时出错：{}".format(traceback.format_exc()))
    wrap_gradio_gpu_call(copy(filename), extra_outputs=None)


def create_embedding(name, initialization_text, nvpt):
    filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    save_pts(filename)
    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    modules.textual_inversion.preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

