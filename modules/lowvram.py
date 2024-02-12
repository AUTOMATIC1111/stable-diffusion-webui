import torch
from torch.utils.weak import WeakIdKeyDictionary
from modules import devices, shared, patches

module_in_gpu = None
cpu = torch.device("cpu")

stream_impl = devices.get_stream_impl()
stream_wrapper = devices.get_stream_wrapper()

class SmartTensorMoverPatches:
    def __init__(self):
        self.memo = WeakIdKeyDictionary()
        self.cleanup_memo = WeakIdKeyDictionary()
        self.model_mover_stream = stream_impl(device=devices.device)

        self.linear_original = patches.patch(__name__, torch.nn.functional, 'linear', self.create_wrapper(torch.nn.functional.linear))
        self.conv2d_original = patches.patch(__name__, torch.nn.functional, 'conv2d', self.create_wrapper(torch.nn.functional.conv2d))
        self.conv3d_original = patches.patch(__name__, torch.nn.functional, 'conv3d', self.create_wrapper(torch.nn.functional.conv3d))
        self.group_norm_original = patches.patch(__name__, torch.nn.functional, 'group_norm', self.create_wrapper(torch.nn.functional.group_norm, type=2))
        self.layer_norm_original = patches.patch(__name__, torch.nn.functional, 'layer_norm', self.create_wrapper(torch.nn.functional.layer_norm, type=2))

    def create_wrapper(self, original, type=1):
        if type == 2:
            def wrapper(input, arg1, weight, bias, *args, **kwargs):
                record_cleanup_weight, record_cleanup_bias = False, False
                current_stream = devices.get_current_stream()

                if weight is not None:
                    new_weight, weight_event = self.memo.get(weight, (None, None))
                    if weight_event is not None:
                        weight = new_weight
                        record_cleanup_weight = True
                        current_stream.wait_event(weight_event)

                if bias is not None:
                    new_bias, bias_event = self.memo.get(bias, (None, None))
                    if bias_event is not None:
                        bias = new_bias
                        record_cleanup_bias = True
                        current_stream.wait_event(bias_event)

                result = original(input, arg1, weight, bias, *args, **kwargs)

                if record_cleanup_weight:
                    self.cleanup_memo[weight] = current_stream.record_event()

                if record_cleanup_bias:
                    self.cleanup_memo[bias] = current_stream.record_event()

                return result
            return wrapper
        else:
            def wrapper(input, weight, bias, *args, **kwargs):
                record_cleanup_weight, record_cleanup_bias = False, False
                current_stream = devices.get_current_stream()

                if weight is not None:
                    new_weight, weight_event = self.memo.get(weight, (None, None))
                    if weight_event is not None:
                        weight = new_weight
                        record_cleanup_weight = True
                        current_stream.wait_event(weight_event)

                if bias is not None:
                    new_bias, bias_event = self.memo.get(bias, (None, None))
                    if bias_event is not None:
                        bias = new_bias
                        record_cleanup_bias = True
                        current_stream.wait_event(bias_event)

                result = original(input, weight, bias, *args, **kwargs)

                if record_cleanup_weight:
                    self.cleanup_memo[weight] = current_stream.record_event()

                if record_cleanup_bias:
                    self.cleanup_memo[bias] = current_stream.record_event()

                return result
            return wrapper

    def __contains__(self, tensor):
        return tensor in self.memo

    def move(self, tensor, device=None):
        device = device or tensor.device
        memo_tensor, memo_event = self.memo.get(tensor, (None, None))

        if memo_tensor is not None:
            return memo_tensor, memo_event

        with stream_wrapper(stream=self.model_mover_stream):
            new_tensor = tensor.to(device=device, copy=True, non_blocking=True)
            new_event = self.model_mover_stream.record_event()
            self.memo[tensor] = (new_tensor, new_event)

        return self.memo[tensor]

    def _forget(self, tensor, tensor_on_device=None):
        if tensor_on_device is not None:
            tensor_used_event = self.cleanup_memo.get(tensor_on_device, None)
            if tensor_used_event is not None:
                self.model_mover_stream.wait_event(tensor_used_event)
                self.cleanup_memo.pop(tensor_on_device, None)
        del self.memo[tensor]

    def forget(self, tensor):
        on_device_tensor = self.memo.get(tensor, None)
        self._forget(tensor, on_device_tensor)

    def forget_batch(self, tensors):
        for tensor in tensors:
            tensor_on_device, _ = self.memo.get(tensor, (None, None))
            if tensor_on_device is not None:
                self._forget(tensor, tensor_on_device)

    def forget_all(self):
        for tensor, (tensor_on_device, _) in self.memo.items():
            if tensor_on_device in self.cleanup_memo:
                self.model_mover_stream.wait_event(
                    self.cleanup_memo[tensor_on_device]
                )
        self.cleanup_memo.clear()
        self.memo.clear()

    def close(self):
        patches.undo(__name__, torch.nn.functional, 'linear')
        patches.undo(__name__, torch.nn.functional, 'conv2d')
        patches.undo(__name__, torch.nn.functional, 'conv3d')
        patches.undo(__name__, torch.nn.functional, 'group_norm')
        patches.undo(__name__, torch.nn.functional, 'layer_norm')

mover = SmartTensorMoverPatches()


class ModelMover:
    @classmethod
    def register(cls, model, max_prefetch=1):
        instance = cls(model, max_prefetch)
        setattr(model, 'lowvram_model_mover', instance)
        return instance

    def __init__(self, model, max_prefetch=1):
        self.model = model
        self.lookahead_distance = max_prefetch
        self.hook_handles = []
        self.submodules_list = self.get_module_list()
        self.submodules_indexer = {}
        self.module_movement_events = {}
        self.default_stream = devices.get_current_stream()
        self.model_mover_stream = stream_impl(device=devices.device)

        for i, module in enumerate(self.submodules_list):
            self.submodules_indexer[module] = i

    def get_module_list(self):
        return []

    def install(self):
        for i in range(len(self.submodules_list)):
            self.hook_handles.append(self.submodules_list[i].register_forward_pre_hook(self._pre_forward_hook))
            self.hook_handles.append(self.submodules_list[i].register_forward_hook(self._post_forward_hook))

    def uninstall(self):
        for handle in self.hook_handles:
            handle.remove()

    def _pre_forward_hook(self, module, _):
        with torch.profiler.record_function("lowvram prehook"):
            with stream_wrapper(stream=self.model_mover_stream):
                idx = self.submodules_indexer[module]
                for i in range(idx, idx + self.lookahead_distance):
                    submodule = self.submodules_list[i % len(self.submodules_list)]
                    if submodule in self.module_movement_events:
                        # already in GPU
                        continue
                    submodule.to(devices.device, non_blocking=True)
                    self.module_movement_events[submodule] = self.model_mover_stream.record_event()

            this_event = self.module_movement_events.get(module, None)
            if this_event is not None:
                self.default_stream.wait_event(this_event)
            else:
                print(f"Module {module.__name__} was not moved to GPU. Taking slow path")
                submodule.to(devices.device, non_blocking=True)

    def _post_forward_hook(self, module, _1, _2):
        with torch.profiler.record_function("lowvram posthook"):
            with stream_wrapper(stream=self.model_mover_stream):
                del self.module_movement_events[module]
                module.to(cpu, non_blocking=True)


class SmartModelMover:
    @classmethod
    def register(cls, model, vram_allowance=0, max_prefetch=10):
        instance = cls(model, vram_allowance, max_prefetch)
        setattr(model, "lowvram_model_mover", instance)
        return instance

    def __init__(self, model, vram_allowance=0, max_prefetch=10):
        self.model = model
        self.vram_allowance = vram_allowance * 1024 * 1024
        self.vram_allowance_remaining = vram_allowance * 1024 * 1024
        self.max_prefetch = max_prefetch
        self.hook_handles = []
        submodules_list = self.get_module_list()
        
        for c in submodules_list:
            c._apply(lambda x: x.pin_memory())

        self.submodules_list = [k for c in submodules_list for k in self.get_childrens(c)]
        self.parameters_list = [[p for p in x.parameters()] for x in self.submodules_list]
        self.parameters_sizes = [sum([p.numel() * p.element_size() for p in x]) for x in self.parameters_list]
        self.online_modules = set()
        self.online_module_count = 0
        self.submodules_indexer = {}

        for i, module in enumerate(self.submodules_list):
            self.submodules_indexer[module] = i

    def test_children(self, op):
        return op.__class__.__name__ in ['Conv2d', 'Conv3d', 'Linear', 'GroupNorm', 'LayerNorm']

    def get_childrens(self, container):
        if isinstance(container, torch.nn.Sequential):
            # return [c for c in container]
            return [cc for c in container for cc in self.get_childrens(c)]
        if 'children' in dir(container):
            childrens = [cc for c in container.children() for cc in self.get_childrens(c)]
            if len(childrens) > 0:
                return childrens
        return [container]

    def drain_allowance(self, idx):
        parameters_len = len(self.parameters_list)

        # no vram limitation is set
        if self.vram_allowance <= 0:
            # fetch up to max_prefetch parameters
            while self.online_module_count < self.max_prefetch:
                param = self.parameters_list[idx]
                self.online_modules.add(idx)
                self.online_module_count += 1
                yield param
                idx = (idx + 1) % parameters_len
            return

        # if there is still vram allowance, and it has not reached max_prefetch
        while self.vram_allowance_remaining > 0 and (self.max_prefetch < 1 or self.online_module_count < self.max_prefetch):
            param = self.parameters_list[idx]
            param_size = self.parameters_sizes[idx]

            # empty module or already online
            if len(param) == 0 or idx in self.online_modules:
                self.online_modules.add(idx)
                self.online_module_count += 1
                idx = (idx + 1) % parameters_len
                continue

            # if the parameter size is bigger than the remaining vram allowance, and there are already online modules
            if (
                param_size > self.vram_allowance_remaining
                and self.online_module_count > 0
            ):
                return
            self.vram_allowance_remaining -= param_size
            self.online_modules.add(idx)
            self.online_module_count += 1
            yield param
            idx = (idx + 1) % parameters_len

    def fill_allowance(self, idx):
        if self.vram_allowance > 0:
            self.vram_allowance_remaining += self.parameters_sizes[idx]
        self.online_modules.remove(idx)
        self.online_module_count -= 1

    def get_module_list(self):
        return []

    def install(self):
        for submodule in self.submodules_list:
            self.hook_handles.append(
                submodule.register_forward_pre_hook(self._pre_forward_hook)
            )
            self.hook_handles.append(
                submodule.register_forward_hook(self._post_forward_hook)
            )

    def uninstall(self):
        for handle in self.hook_handles:
            handle.remove()
        
        for idx in self.online_modules:
            mover.forget_batch(self.parameters_list[idx])

    def preload(self):
        idx = 0
        for parameters in self.drain_allowance(idx):
            for param in parameters:
                mover.move(param, device=devices.device)


    def _pre_forward_hook(self, module, *args, **kwargs):
        idx = self.submodules_indexer[module]
        for parameters in self.drain_allowance(idx):
            for param in parameters:
                mover.move(param, device=devices.device)

    def _post_forward_hook(self, module, *args, **kwargs):
        idx = self.submodules_indexer[module]

        mover.forget_batch(self.parameters_list[idx])
        self.fill_allowance(idx)


class DiffModelMover(SmartModelMover):
    def get_module_list(self):
        modules = []
        modules.append(self.model.time_embed)
        for block in self.model.input_blocks:
            modules.append(block)
        modules.append(self.model.middle_block)
        for block in self.model.output_blocks:
            modules.append(block)
        return modules


def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None

    mover.forget_all()


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
        mp = DiffModelMover.register(diff_model, max_prefetch=70)
        mp.install()
        mp.preload()

        # diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        # for block in diff_model.input_blocks:
        #     block.register_forward_pre_hook(send_me_to_gpu)
        # diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        # for block in diff_model.output_blocks:
        #     block.register_forward_pre_hook(send_me_to_gpu)


def is_enabled(sd_model):
    return sd_model.lowvram
