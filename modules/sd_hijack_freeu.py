import math
import functools
import torch
from modules import shared

# based on <https://github.com/ljleb/sd-webui-freeu/blob/main/lib_free_u/unet.py>
# official params are b1,b2,s1,s2

# extra params that can be made configurable if needed are:
backbone_width = 0.5
backbone_offset = 0.0
skip_cutoff = 0.0
skip_high_end_factor = 1.0
start_ratio = 0.0
stop_ratio = 1.0
transition_smoothness = 0.0

# internal state
state_enabled = False
cat_original = None


def to_denoising_step(number, steps=None) -> int:
    if steps is None:
        steps = shared.state.sampling_steps
    if isinstance(number, float):
        return int(number * steps)
    return number


def get_schedule_ratio():
    start_step = to_denoising_step(start_ratio)
    stop_step = to_denoising_step(stop_ratio)
    if start_step == stop_step:
        smooth_schedule_ratio = 0.0
    elif shared.state.sampling_step < start_step:
        smooth_schedule_ratio = min(1.0, max(0.0, shared.state.sampling_step / start_step))
    else:
        smooth_schedule_ratio = min(1.0, max(0.0, 1 + (shared.state.sampling_step - start_step) / (start_step - stop_step)))
    flat_schedule_ratio = 1.0 if start_step <= shared.state.sampling_step < stop_step else 0.0
    return lerp(flat_schedule_ratio, smooth_schedule_ratio, transition_smoothness)


def lerp(a, b, r):
    return (1-r)*a + r*b


def free_u_cat_hijack(hs, *args, original_function, **kwargs):
    if not shared.opts.freeu_enabled:
        return original_function(hs, *args, **kwargs)
    schedule_ratio = get_schedule_ratio()
    if schedule_ratio == 0:
        return original_function(hs, *args, **kwargs)
    try:
        h, h_skip = hs
        if list(kwargs.keys()) != ["dim"] or kwargs.get("dim", -1) != 1:
            return original_function(hs, *args, **kwargs)
    except ValueError:
        return original_function(hs, *args, **kwargs)
    dims = h.shape[1]
    if dims not in [1280, 640, 320]:
        return original_function(hs, *args, **kwargs)
    index = [1280, 640, 320].index(dims)
    if index > 1: # not 1st or 2nd stage
        return original_function([h, h_skip], *args, **kwargs)
    region_begin, region_end, region_inverted = ratio_to_region(backbone_width, backbone_offset, dims)
    mask = torch.arange(dims)
    mask = (region_begin <= mask) & (mask <= region_end)
    if region_inverted:
        mask = ~mask
    backbone_factor = shared.opts.freeu_b1 if index == 0 else shared.opts.freeu_b2
    skip_factor = shared.opts.freeu_s1 if index == 0 else shared.opts.freeu_s2
    h[:, mask] *= lerp(1, backbone_factor, schedule_ratio)
    h_skip = filter_skip(h_skip, threshold=skip_cutoff, scale=lerp(1, skip_factor, schedule_ratio), scale_high=lerp(1, skip_high_end_factor, schedule_ratio))
    return original_function([h, h_skip], *args, **kwargs)


def no_gpu_complex_support():
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    try:
        import torch_directml
    except ImportError:
        dml_available = False
    else:
        dml_available = torch_directml.is_available()
    return mps_available or dml_available


def filter_skip(x, threshold, scale, scale_high):
    if scale == 1 and scale_high == 1:
        return x
    fft_device = x.device
    if no_gpu_complex_support():
        fft_device = "cpu"
    # FFT
    x_freq = torch.fft.fftn(x.to(fft_device).float(), dim=(-2, -1)) # pylint: disable=E1102
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1)) # pylint: disable=E1102
    B, C, H, W = x_freq.shape
    mask = torch.full((B, C, H, W), float(scale_high), device=fft_device)
    crow, ccol = H // 2, W // 2
    threshold_row = max(1, math.floor(crow * threshold))
    threshold_col = max(1, math.floor(ccol * threshold))
    mask[..., crow - threshold_row:crow + threshold_row, ccol - threshold_col:ccol + threshold_col] = scale
    x_freq *= mask
    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1)) # pylint: disable=E1102
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real.to(device=x.device, dtype=x.dtype) # pylint: disable=E1102
    return x_filtered


def ratio_to_region(width: float, offset: float, n: int):
    if width < 0:
        offset += width
        width = -width
    width = min(width, 1)
    if offset < 0:
        offset = 1 + offset - int(offset)
    offset = math.fmod(offset, 1.0)
    if width + offset <= 1:
        inverted = False
        start = offset * n
        end = (width + offset) * n
    else:
        inverted = True
        start = (width + offset - 1) * n
        end = offset * n
    return round(start), round(end), inverted


def apply_freeu(p, backend_original):
    from modules.sd_hijack_unet import th
    global state_enabled # pylint: disable=global-statement
    global cat_original # pylint: disable=global-statement
    if backend_original:
        if shared.opts.freeu_enabled:
            p.extra_generation_params['FreeU'] = f'b1={shared.opts.freeu_b1} b2={shared.opts.freeu_b2} s1={shared.opts.freeu_s1} s2={shared.opts.freeu_s2}'
            if not state_enabled: # otherwise already patched
                cat_original = th.cat
                th.cat = functools.partial(free_u_cat_hijack, original_function=th.cat)
                state_enabled = True
        else:
            if cat_original is not None:
                th.cat = cat_original
                state_enabled = False
    elif hasattr(p.sd_model, 'enable_freeu'):
        if shared.opts.freeu_enabled:
            p.extra_generation_params['FreeU'] = f'b1={shared.opts.freeu_b1} b2={shared.opts.freeu_b2} s1={shared.opts.freeu_s1} s2={shared.opts.freeu_s2}'
            p.sd_model.enable_freeu(s1=shared.opts.freeu_s1, s2=shared.opts.freeu_s2, b1=shared.opts.freeu_b1, b2=shared.opts.freeu_b2)
            state_enabled = True
        elif state_enabled:
            p.sd_model.disable_freeu()
            state_enabled = False
    if shared.opts.freeu_enabled:
        shared.log.info(f'Applying free-u: b1={shared.opts.freeu_b1} b2={shared.opts.freeu_b2} s1={shared.opts.freeu_s1} s2={shared.opts.freeu_s2}')
