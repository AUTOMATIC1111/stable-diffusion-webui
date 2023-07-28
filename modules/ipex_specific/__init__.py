import os
import torch
import intel_extension_for_pytorch as ipex
from modules import shared
from modules.sd_hijack_utils import CondFunc

def ipex_no_cuda(orig_func, *args, **kwargs): # pylint: disable=redefined-outer-name
    torch.cuda.is_available = lambda: False
    orig_func(*args, **kwargs)
    torch.cuda.is_available = torch.xpu.is_available

def ipex_init():
    #Fix functions with ipex
    torch.cuda.is_available = torch.xpu.is_available
    torch.cuda.device = torch.xpu.device
    torch.cuda.device_count = torch.xpu.device_count
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.get_device_name = torch.xpu.get_device_name
    torch.cuda.get_device_properties = torch.xpu.get_device_properties
    torch._utils._get_available_device_type = lambda: "xpu" # pylint: disable=protected-access
    torch.cuda.set_device = torch.xpu.set_device
    torch.cuda.synchronize = torch.xpu.synchronize
    torch.Tensor.cuda = torch.Tensor.xpu

    torch.xpu.empty_cache = torch.xpu.empty_cache if "WSL2" not in os.popen("uname -a").read() else lambda: None
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.ipc_collect = lambda: None
    torch.cuda.memory_stats = torch.xpu.memory_stats
    torch.cuda.mem_get_info = lambda device=None: [(torch.xpu.get_device_properties(device).total_memory - torch.xpu.memory_allocated(device)), torch.xpu.get_device_properties(device).total_memory]
    torch.cuda.memory_allocated = torch.xpu.memory_allocated
    torch.cuda.max_memory_allocated = torch.xpu.max_memory_allocated
    torch.cuda.reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
    torch.cuda.utilization = lambda: 0

    torch.cuda.get_rng_state_all = torch.xpu.get_rng_state_all
    torch.cuda.set_rng_state_all = torch.xpu.set_rng_state_all
    try:
        torch.cuda.amp.GradScaler = torch.xpu.amp.GradScaler
    except Exception:
        torch.cuda.amp.GradScaler = ipex.cpu.autocast._grad_scaler.GradScaler

    #Adetailer and more:
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format:
        orig_func(self, shared.device, dtype=dtype, non_blocking=non_blocking, copy=copy, memory_format=memory_format),
        lambda orig_func, self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format:
        (type(device) is torch.device and device.type == "cuda") or (type(device) is str and "cuda" in device))
    CondFunc('torch.empty',
        lambda orig_func, *args, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format:
        orig_func(*args, out=out, dtype=dtype, layout=layout, device=shared.device, requires_grad=requires_grad, pin_memory=pin_memory, memory_format=memory_format),
        lambda orig_func, *args, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format:
        (type(device) is torch.device and device.type == "cuda") or (type(device) is str and "cuda" in device))

    #Broken functions when torch.cuda.is_available is True:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    #Functions with dtype errors:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    #Diffusers bfloat16:
    CondFunc('torch.nn.modules.Conv2d._conv_forward',
        lambda orig_func, self, input, weight, bias=None: orig_func(self, input.to(weight.data.dtype), weight, bias=bias),
        lambda orig_func, self, input, weight, bias=None: input.dtype != weight.data.dtype)

    #Functions that does not work with the XPU:
    #UniPC:
    CondFunc('torch.linalg.solve',
        lambda orig_func, A, B, *args, left=True, out=None: orig_func(A.to("cpu"), B.to("cpu"), *args, left=left, out=out).to(shared.device),
        lambda orig_func, A, B, *args, left=True, out=None: A.device != torch.device("cpu") or B.device != torch.device("cpu"))
    #SDE Samplers:
    CondFunc('torch.Generator',
        lambda orig_func, device: torch.xpu.Generator(device),
        lambda orig_func, device: device != torch.device("cpu") and device != "cpu")
    #Latent antialias:
    CondFunc('torch.nn.functional.interpolate',
        lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False:
        orig_func(input.to("cpu"), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(shared.device),
        lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: antialias)
    #Diffusers Float64 (ARC GPUs doesn't support double or Float64):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
            lambda orig_func, ndarray: orig_func(ndarray.astype('float32')),
            lambda orig_func, ndarray: ndarray.dtype == float)
    #ControlNet and TiledVAE:
    CondFunc('torch.batch_norm',
        lambda orig_func, input, weight=None, bias=None, running_mean=None, running_var=None, training=False, momentum=0.1, eps=1e-5, cudnn_enabled=True: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=shared.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=shared.device),
        running_mean, running_var, training, momentum, eps, cudnn_enabled),
        lambda orig_func, input, weight=None, bias=None, running_mean=None, running_var=None, training=False, momentum=0.1, eps=1e-5, cudnn_enabled=True: input.device != torch.device("cpu"))
    #ControlNet
    CondFunc('torch.instance_norm',
        lambda orig_func, input, weight=None, bias=None, running_mean=None, running_var=None, use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=True: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=shared.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=shared.device),
        running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled),
        lambda orig_func, input, weight=None, bias=None, running_mean=None, running_var=None, use_input_stats=True, momentum=0.1, eps=1e-5, cudnn_enabled=True: input.device != torch.device("cpu"))
