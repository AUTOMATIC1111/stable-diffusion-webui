" Memory profiling callbacks "

import tracemalloc, threading, torch, time
from ..utils.mem import *
from ..basic_train import *
from ..torch_core import *
from ..utils.pynvml_gate import *

if use_gpu: pynvml = load_pynvml_env()

class PeakMemMetric(LearnerCallback):
    "Callback that measures used and peaked general and GPU memory."

    _order=-20 # Needs to run before the recorder

    def __init__(self, learn:Learner):
        super().__init__(learn)
        assert torch.cuda.is_available(), "pytorch CUDA is required"
        preload_pytorch()

    def peak_monitor_start(self):
        self.peak_monitoring = True

        # start RAM tracing
        tracemalloc.start()

        # this thread samples RAM usage as long as the current epoch of the fit loop is running
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def peak_monitor_stop(self):
        tracemalloc.stop()
        self.peak_monitoring = False

    def peak_monitor_func(self):
        self.gpu_mem_used_peak = -1

        gpu_id = torch.cuda.current_device()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        while True:
            gpu_mem_used = gpu_mem_get_used_fast(gpu_handle)
            self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)
            if not self.peak_monitoring: break
            time.sleep(0.001) # 1msec

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['cpu used',  'peak', 'gpu used',  'peak'])

    def on_epoch_begin(self, **kwargs):
        self.peak_monitor_start()
        self.gpu_before = gpu_mem_get_used_no_cache()

    def on_epoch_end(self, last_metrics, **kwargs):
        cpu_used, cpu_peak =  list(map(lambda x: int(x/2**20), tracemalloc.get_traced_memory()))
        self.peak_monitor_stop()
        gpu_used = gpu_mem_get_used_no_cache() - self.gpu_before
        gpu_peak = self.gpu_mem_used_peak      - self.gpu_before
        # can be negative, due to unreliable peak monitor thread
        if gpu_peak < 0:   gpu_peak = 0
        # since we want the overhead only, subtract delta used if it's positive
        elif gpu_used > 0: gpu_peak -= gpu_used
        # The numbers are deltas in MBs (beginning of the epoch and the end)
        return add_metrics(last_metrics, [cpu_used, cpu_peak, gpu_used, gpu_peak])
