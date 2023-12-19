import contextlib
import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
from modules.sd_hijack_utils import CondFunc
from modules import devices

# pylint: disable=protected-access, missing-function-docstring, line-too-long, unnecessary-lambda, no-else-return

def _shutdown_workers(self):
    if torch.utils.data._utils is None or torch.utils.data._utils.python_exit_status is True or torch.utils.data._utils.python_exit_status is None:
        return
    if hasattr(self, "_shutdown") and not self._shutdown:
        self._shutdown = True
        try:
            if hasattr(self, '_pin_memory_thread'):
                self._pin_memory_thread_done_event.set()
                self._worker_result_queue.put((None, None))
                self._pin_memory_thread.join()
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()
            self._workers_done_event.set()
            for worker_id in range(len(self._workers)):
                if self._persistent_workers or self._workers_status[worker_id]:
                    self._mark_worker_as_unavailable(worker_id, shutdown=True)
            for w in self._workers: # pylint: disable=invalid-name
                w.join(timeout=torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL)
            for q in self._index_queues: # pylint: disable=invalid-name
                q.cancel_join_thread()
                q.close()
        finally:
            if self._worker_pids_set:
                torch.utils.data._utils.signal_handling._remove_worker_pids(id(self))
                self._worker_pids_set = False
            for w in self._workers: # pylint: disable=invalid-name
                if w.is_alive():
                    w.terminate()

class DummyDataParallel(torch.nn.Module): # pylint: disable=missing-class-docstring, unused-argument, too-few-public-methods
    def __new__(cls, module, device_ids=None, output_device=None, dim=0): # pylint: disable=unused-argument
        if isinstance(device_ids, list) and len(device_ids) > 1:
            print("IPEX backend doesn't support DataParallel on multiple XPU devices")
        return module.to(devices.device)

def return_null_context(*args, **kwargs): # pylint: disable=unused-argument
    return contextlib.nullcontext()

def check_device(device):
    return bool((isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def return_xpu(device):
    return f"xpu:{device.split(':')[-1]}" if isinstance(device, str) and ":" in device else f"xpu:{device}" if isinstance(device, int) else torch.device(devices.device) if isinstance(device, torch.device) else devices.device

def ipex_no_cuda(orig_func, *args, **kwargs):
    torch.cuda.is_available = lambda: False
    orig_func(*args, **kwargs)
    torch.cuda.is_available = torch.xpu.is_available

original_autocast = torch.autocast
def ipex_autocast(*args, **kwargs):
    if len(args) > 0 and args[0] == "cuda" or args[0] == "xpu":
        if "dtype" in kwargs:
            return original_autocast("xpu", *args[1:], **kwargs)
        else:
            return original_autocast("xpu", *args[1:], dtype=devices.dtype, **kwargs)
    else:
        return original_autocast(*args, **kwargs)

# Embedding BF16
original_torch_cat = torch.cat
def torch_cat(tensor, *args, **kwargs):
    if len(tensor) == 3 and (tensor[0].dtype != tensor[1].dtype or tensor[2].dtype != tensor[1].dtype):
        return original_torch_cat([tensor[0].to(tensor[1].dtype), tensor[1], tensor[2].to(tensor[1].dtype)], *args, **kwargs)
    else:
        return original_torch_cat(tensor, *args, **kwargs)

# Latent antialias:
original_interpolate = torch.nn.functional.interpolate
def interpolate(tensor, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False): # pylint: disable=too-many-arguments
    if antialias or align_corners is not None:
        return_device = tensor.device
        return_dtype = tensor.dtype
        return original_interpolate(tensor.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(tensor, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)

original_linalg_solve = torch.linalg.solve
def linalg_solve(A, B, *args, **kwargs): # pylint: disable=invalid-name
    if A.device != torch.device("cpu") or B.device != torch.device("cpu"):
        return_device = A.device
        return original_linalg_solve(A.to("cpu"), B.to("cpu"), *args, **kwargs).to(return_device)
    else:
        return original_linalg_solve(A, B, *args, **kwargs)

if torch.xpu.has_fp64_dtype():
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
else:
    # 64 bit attention workarounds for Alchemist:
    try:
        from .attention import torch_bmm_32_bit as original_torch_bmm
        from .attention import scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention
    except Exception: # pylint: disable=broad-exception-caught
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

# dtype errors:
def torch_bmm(input, mat2, *, out=None):
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)
    return original_torch_bmm(input, mat2, out=out)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    if query.dtype != key.dtype:
        key = key.to(dtype=query.dtype)
    if query.dtype != value.dtype:
        value = value.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

@property
def is_cuda(self):
    return self.device.type == 'xpu'

def ipex_hijacks():
    CondFunc('torch.tensor',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.Tensor.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.UntypedStorage.__init__',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.UntypedStorage.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.empty',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.randn',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.ones',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.zeros',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.linspace',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.load',
        lambda orig_func, f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kwargs:
        orig_func(orig_func, f, map_location=return_xpu(map_location), pickle_module=pickle_module, weights_only=weights_only, mmap=mmap, **kwargs),
        lambda orig_func, f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kwargs: check_device(map_location))
    if hasattr(torch.xpu, "Generator"):
        CondFunc('torch.Generator',
            lambda orig_func, device=None: torch.xpu.Generator(return_xpu(device)),
            lambda orig_func, device=None: device is not None and device != torch.device("cpu") and device != "cpu")
    else:
        CondFunc('torch.Generator',
            lambda orig_func, device=None: orig_func(return_xpu(device)),
            lambda orig_func, device=None: check_device(device))

    # TiledVAE and ControlNet:
    CondFunc('torch.batch_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=input.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=input.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))
    CondFunc('torch.instance_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=input.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=input.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))

    # Functions with dtype errors:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    # Training:
    CondFunc('torch.nn.modules.linear.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.conv.Conv2d.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    # BF16:
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)
    # SwinIR BF16:
    CondFunc('torch.nn.functional.pad',
        lambda orig_func, input, pad, mode='constant', value=None: orig_func(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16),
        lambda orig_func, input, pad, mode='constant', value=None: mode == 'reflect' and input.dtype == torch.bfloat16)

    # Diffusers Float64 (Alchemist GPUs doesn't support 64 bit):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
        lambda orig_func, ndarray: orig_func(ndarray.astype('float32')),
        lambda orig_func, ndarray: ndarray.dtype == float)

    # Broken functions when torch.cuda.is_available is True:
    # Pin Memory:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    # Functions that make compile mad with CondFunc:
    torch.nn.DataParallel = DummyDataParallel
    torch.utils.data.dataloader._MultiProcessingDataLoaderIter._shutdown_workers = _shutdown_workers

    torch.autocast = ipex_autocast
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.UntypedStorage.is_cuda = is_cuda

    torch.nn.functional.interpolate = interpolate
    torch.linalg.solve = linalg_solve

    torch.bmm = torch_bmm
    torch.cat = torch_cat
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
