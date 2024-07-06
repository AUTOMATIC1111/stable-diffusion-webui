import sys
import contextlib
from functools import lru_cache

import torch
from modules import errors, shared, npu_specific

if sys.platform == "darwin":
    from modules import mac_specific

if shared.cmd_opts.use_ipex:
    from modules import xpu_specific


def has_xpu() -> bool:
    return shared.cmd_opts.use_ipex and xpu_specific.has_xpu


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def cuda_no_autocast(device_id=None) -> bool:
    if device_id is None:
        device_id = get_cuda_device_id()
    return (
        torch.cuda.get_device_capability(device_id) == (7, 5)
        and torch.cuda.get_device_name(device_id).startswith("NVIDIA GeForce GTX 16")
    )


def get_cuda_device_id():
    return (
        int(shared.cmd_opts.device_id)
        if shared.cmd_opts.device_id is not None and shared.cmd_opts.device_id.isdigit()
        else 0
    ) or torch.cuda.current_device()


def get_cuda_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    if has_xpu():
        return xpu_specific.get_xpu_device_string()

    if npu_specific.has_npu:
        return npu_specific.get_npu_device_string()

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu or "all" in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()

    if has_xpu():
        xpu_specific.torch_xpu_gc()

    if npu_specific.has_npu:
        torch_npu_set_device()
        npu_specific.torch_npu_gc()


def torch_npu_set_device():
    # Work around due to bug in torch_npu, revert me after fixed, @see https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue
    if npu_specific.has_npu:
        torch.npu.set_device(0)


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if cuda_no_autocast():
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu: torch.device = torch.device("cpu")
fp8: bool = False
# Force fp16 for all models in inference. No casting during inference.
# This flag is controlled by "--precision half" command line arg.
force_fp16: bool = False
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
dtype_inference: torch.dtype = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    if force_fp16:
        return input.to(torch.float16)
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


nv_rng = None
patch_module_list = [
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MultiheadAttention,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
]


def manual_cast_forward(target_dtype):
    def forward_wrapper(self, *args, **kwargs):
        if any(
            isinstance(arg, torch.Tensor) and arg.dtype != target_dtype
            for arg in args
        ):
            args = [arg.to(target_dtype) if isinstance(arg, torch.Tensor) else arg for arg in args]
            kwargs = {k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        org_dtype = target_dtype
        for param in self.parameters():
            if param.dtype != target_dtype:
                org_dtype = param.dtype
                break

        if org_dtype != target_dtype:
            self.to(target_dtype)
        result = self.org_forward(*args, **kwargs)
        if org_dtype != target_dtype:
            self.to(org_dtype)

        if target_dtype != dtype_inference:
            if isinstance(result, tuple):
                result = tuple(
                    i.to(dtype_inference)
                    if isinstance(i, torch.Tensor)
                    else i
                    for i in result
                )
            elif isinstance(result, torch.Tensor):
                result = result.to(dtype_inference)
        return result
    return forward_wrapper


@contextlib.contextmanager
def manual_cast(target_dtype):
    applied = False
    for module_type in patch_module_list:
        if hasattr(module_type, "org_forward"):
            continue
        applied = True
        org_forward = module_type.forward
        if module_type == torch.nn.MultiheadAttention:
            module_type.forward = manual_cast_forward(torch.float32)
        else:
            module_type.forward = manual_cast_forward(target_dtype)
        module_type.org_forward = org_forward
    try:
        yield None
    finally:
        if applied:
            for module_type in patch_module_list:
                if hasattr(module_type, "org_forward"):
                    module_type.forward = module_type.org_forward
                    delattr(module_type, "org_forward")


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if force_fp16:
        # No casting during inference if force_fp16 is enabled.
        # All tensor dtype conversion happens before inference.
        return contextlib.nullcontext()

    if fp8 and device==cpu:
        return torch.autocast("cpu", dtype=torch.bfloat16, enabled=True)

    if fp8 and dtype_inference == torch.float32:
        return manual_cast(dtype)

    if dtype == torch.float32 or dtype_inference == torch.float32:
        return contextlib.nullcontext()

    if has_xpu() or has_mps() or cuda_no_autocast():
        return manual_cast(dtype)

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.isnan(x[(0, ) * len(x.shape)]):
        return

    if where == "unet":
        message = "A tensor with NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocates about 700MB of memory and
    spends about 2.7 seconds doing that, at least with NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)


def force_model_fp16():
    """
    ldm and sgm has modules.diffusionmodules.util.GroupNorm32.forward, which
    force conversion of input to float32. If force_fp16 is enabled, we need to
    prevent this casting.
    """
    assert force_fp16
    import sgm.modules.diffusionmodules.util as sgm_util
    import ldm.modules.diffusionmodules.util as ldm_util
    sgm_util.GroupNorm32 = torch.nn.GroupNorm
    ldm_util.GroupNorm32 = torch.nn.GroupNorm
    print("ldm/sgm GroupNorm32 replaced with normal torch.nn.GroupNorm due to `--precision half`.")
