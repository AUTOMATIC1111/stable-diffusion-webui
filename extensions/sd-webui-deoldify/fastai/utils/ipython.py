"ipython utils"

import os, functools, traceback, gc

def is_in_ipython():
    "Is the code running in the ipython environment (jupyter including)"

    program_name = os.path.basename(os.getenv('_', ''))

    if ('jupyter-notebook' in program_name or # jupyter-notebook
        'ipython'          in program_name or # ipython
        'JPY_PARENT_PID'   in os.environ):    # ipython-notebook
        return True
    else:
        return False

IS_IN_IPYTHON = is_in_ipython()

def is_in_colab():
    "Is the code running in Google Colaboratory?"
    if not IS_IN_IPYTHON: return False
    try:
        from google import colab
        return True
    except: return False

IS_IN_COLAB = is_in_colab()

def get_ref_free_exc_info():
    "Free traceback from references to locals() in each frame to avoid circular reference leading to gc.collect() unable to reclaim memory"
    type, val, tb = sys.exc_info()
    traceback.clear_frames(tb)
    return (type, val, tb)

def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tb_clear_frames = os.environ.get('FASTAI_TB_CLEAR_FRAMES', None)
        if not IS_IN_IPYTHON or tb_clear_frames=="0":
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            if ("CUDA out of memory" in str(e) or
                "device-side assert triggered" in str(e) or
                tb_clear_frames == "1"):
                type, val, tb = get_ref_free_exc_info() # must!
                gc.collect()
                if "device-side assert triggered" in str(e):
                    warn("""When 'device-side assert triggered' error happens, it's not possible to recover and you must restart the kernel to continue. Use os.environ['CUDA_LAUNCH_BLOCKING']="1" before restarting to debug""")
                raise type(val).with_traceback(tb) from None
            else: raise # re-raises the exact last exception
    return wrapper

class gpu_mem_restore_ctx():
    "context manager to reclaim RAM if an exception happened under ipython"
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val: return True
        traceback.clear_frames(exc_tb)
        gc.collect()
        raise exc_type(exc_val).with_traceback(exc_tb) from None
