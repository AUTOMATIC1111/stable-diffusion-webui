from contextlib import contextmanager, nullcontext
import torch
from modules import devices, shared, patches

module_in_gpu = None
cpu = torch.device("cpu")

stream_impl = devices.get_stream_impl()
stream_wrapper = devices.get_stream_wrapper()


use_streamlined_lowvram = torch.cuda.is_available() and not shared.opts.use_non_streamlined_lowvram and stream_impl is not None and stream_wrapper is not None


def is_same_device(device1, device2):
    tensor1_device_type = device1.type
    tensor2_device_type = device2.type
    tensor1_device_index = device1.index or 0
    tensor2_device_index = device2.index or 0
    return (
        tensor1_device_type == tensor2_device_type
        and tensor1_device_index == tensor2_device_index
    )

class RTTensorMoverPatches:
    def __init__(self):
        self.mover_stream = stream_impl(device=devices.device)
        self.calc_stream = stream_impl(device=devices.device)
        self.stash = {}
        self.speed_limit_loop_head = 0
        self.speed_limit_loop = []


        self.linear_original = patches.patch(
            __name__,
            torch.nn.functional,
            "linear",
            self._wrapper_default(torch.nn.functional.linear),
        )
        self.conv2d_original = patches.patch(
            __name__,
            torch.nn.functional,
            "conv2d",
            self._wrapper_default(torch.nn.functional.conv2d),
        )
        self.conv3d_original = patches.patch(
            __name__,
            torch.nn.functional,
            "conv3d",
            self._wrapper_default(torch.nn.functional.conv3d),
        )
        self.group_norm_original = patches.patch(
            __name__,
            torch.nn.functional,
            "group_norm",
            self._wrapper_group_norm(torch.nn.functional.group_norm),
        )
        self.layer_norm_original = patches.patch(
            __name__,
            torch.nn.functional,
            "layer_norm",
            self._wrapper_layer_norm(torch.nn.functional.layer_norm),
        )

    @contextmanager
    def wrap_weight_biases(self, input, weight, bias):
        if not is_same_device(input.device, devices.device):
            yield (weight, bias)
            return

        moved = False
        before_calc_event, after_calc_event = None, None
        with stream_wrapper(stream=self.mover_stream):
            if weight is not None and not is_same_device(weight.device, input.device):
                weight = weight.to(device=input.device, copy=True, non_blocking=weight.is_pinned())
                moved = True
            if bias is not None and not is_same_device(bias.device, input.device):
                bias = bias.to(device=input.device, copy=True, non_blocking=bias.is_pinned())
                moved = True
            before_calc_event = self.mover_stream.record_event()

        if not moved:
            yield (weight, bias)
            return

        with stream_wrapper(stream=self.calc_stream):
            if before_calc_event is not None:
                self.calc_stream.wait_event(before_calc_event)
            yield (weight, bias)
            after_calc_event = self.calc_stream.record_event()
            self.stash[id(after_calc_event)] = (weight, bias, after_calc_event)

        to_remove = []
        for k, (_, _, e) in self.stash.items():
            if e.query():
                to_remove.append(k)

        for k in to_remove:
            del self.stash[k]

        if len(self.speed_limit_loop) < shared.opts.lowvram_max_loaded_module:
            self.speed_limit_loop.extend([None] * (shared.opts.lowvram_max_loaded_module - len(self.speed_limit_loop)))

        self.speed_limit_loop[self.speed_limit_loop_head] = after_calc_event
        self.speed_limit_loop_head = (self.speed_limit_loop_head + 1) % shared.opts.lowvram_max_loaded_module
        if self.speed_limit_loop[self.speed_limit_loop_head] is not None:
            self.mover_stream.wait_event(self.speed_limit_loop[self.speed_limit_loop_head])

    def _wrapper_default(self, original):
        def wrapper(input, weight, bias=None, *args, **kwargs):
            with self.wrap_weight_biases(input, weight, bias) as (w, b):
                return original(input, w, b, *args, **kwargs)
        return wrapper

    def _wrapper_group_norm(self, original):
        def wrapper(input, num_groups, weight=None, bias=None, *args, **kwargs):
            with self.wrap_weight_biases(input, weight, bias) as (w, b):
                return original(input, num_groups, w, b, *args, **kwargs)
        return wrapper

    def _wrapper_layer_norm(self, original):
        def wrapper(input, normalized_shape, weight=None, bias=None, *args, **kwargs):
            with self.wrap_weight_biases(input, weight, bias) as (w, b):
                return original(input, normalized_shape, w, b, *args, **kwargs)
        return wrapper

    def close(self):
        patches.undo(__name__, torch.nn.functional, "linear")
        patches.undo(__name__, torch.nn.functional, "conv2d")
        patches.undo(__name__, torch.nn.functional, "conv3d")
        patches.undo(__name__, torch.nn.functional, "group_norm")
        patches.undo(__name__, torch.nn.functional, "layer_norm")


rtmover = None
if use_streamlined_lowvram:
    rtmover = RTTensorMoverPatches()


def calc_wrapper():
    if rtmover is not None:
        return stream_wrapper(stream=rtmover.calc_stream)
    return nullcontext()


def calc_sync():
    if rtmover is not None:
        return rtmover.calc_stream.synchronize()
    return nullcontext()


def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None


def is_needed(sd_model):
    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or shared.cmd_opts.medvram_sdxl and hasattr(sd_model, 'conditioner')


def apply(sd_model):
    enable = is_needed(sd_model)
    shared.parallel_processing_allowed = not enable

    if enable:
        setup_for_low_vram(sd_model, not shared.cmd_opts.lowvram)
    else:
        sd_model.lowvram = False


def setup_for_low_vram(sd_model, use_medvram):
    if getattr(sd_model, 'lowvram', False):
        return

    sd_model.lowvram = True

    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(devices.device)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods

    first_stage_model = sd_model.first_stage_model
    first_stage_model_encode = sd_model.first_stage_model.encode
    first_stage_model_decode = sd_model.first_stage_model.decode

    def first_stage_model_encode_wrap(x):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_encode(x)

    def first_stage_model_decode_wrap(z):
        send_me_to_gpu(first_stage_model, None)
        return first_stage_model_decode(z)

    to_remain_in_cpu = [
        (sd_model, 'first_stage_model'),
        (sd_model, 'depth_model'),
        (sd_model, 'embedder'),
        (sd_model, 'model'),
        (sd_model, 'embedder'),
    ]

    is_sdxl = hasattr(sd_model, 'conditioner')
    is_sd2 = not is_sdxl and hasattr(sd_model.cond_stage_model, 'model')

    if is_sdxl:
        to_remain_in_cpu.append((sd_model, 'conditioner'))
    elif is_sd2:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'model'))
    else:
        to_remain_in_cpu.append((sd_model.cond_stage_model, 'transformer'))

    # remove several big modules: cond, first_stage, depth/embedder (if applicable), and unet from the model
    stored = []
    for obj, field in to_remain_in_cpu:
        module = getattr(obj, field, None)
        stored.append(module)
        setattr(obj, field, None)

    # send the model to GPU.
    sd_model.to(devices.device)

    # put modules back. the modules will be in CPU.
    for (obj, field), module in zip(to_remain_in_cpu, stored):
        setattr(obj, field, module)

    # register hooks for those the first three models
    if is_sdxl:
        sd_model.conditioner.register_forward_pre_hook(send_me_to_gpu)
    elif is_sd2:
        sd_model.cond_stage_model.model.register_forward_pre_hook(send_me_to_gpu)
        sd_model.cond_stage_model.model.token_embedding.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.model] = sd_model.cond_stage_model
        parents[sd_model.cond_stage_model.model.token_embedding] = sd_model.cond_stage_model
    else:
        sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
        parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = first_stage_model_encode_wrap
    sd_model.first_stage_model.decode = first_stage_model_decode_wrap
    if sd_model.depth_model:
        sd_model.depth_model.register_forward_pre_hook(send_me_to_gpu)
    if sd_model.embedder:
        sd_model.embedder.register_forward_pre_hook(send_me_to_gpu)

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(devices.device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model

        if use_streamlined_lowvram:
            # put it into pinned memory to achieve data transfer overlap
            diff_model.time_embed._apply(lambda x: x.pin_memory(device=devices.device))
            for block in diff_model.input_blocks:
                block._apply(lambda x: x.pin_memory(device=devices.device))
            diff_model.middle_block._apply(lambda x: x.pin_memory(device=devices.device))
            for block in diff_model.output_blocks:
                block._apply(lambda x: x.pin_memory(device=devices.device))
        else:
            diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
            for block in diff_model.input_blocks:
                block.register_forward_pre_hook(send_me_to_gpu)
            diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
            for block in diff_model.output_blocks:
                block.register_forward_pre_hook(send_me_to_gpu)


def is_enabled(sd_model):
    return sd_model.lowvram
