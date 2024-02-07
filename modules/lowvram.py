import torch
from modules import devices, shared

module_in_gpu = None
cpu = torch.device("cpu")

stream_impl = devices.get_stream_impl()
stream_wrapper = devices.get_stream_wrapper()

class ModelMover:
    @classmethod
    def register(cls, model, lookahead_distance=1):
        instance = cls(model, lookahead_distance)
        setattr(model, 'lowvram_model_mover', instance)
        return instance

    def __init__(self, model, lookahead_distance=1):
        self.model = model
        self.lookahead_distance = lookahead_distance
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
        with stream_wrapper(stream=self.model_mover_stream):
            del self.module_movement_events[module]
            module.to(cpu, non_blocking=True)


class DiffModelMover(ModelMover):
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

        try:
            name = module._get_name()
        except:
            try:
                name = module.__name__
            except:
                try:
                    name = module.__class__.__name__
                except:
                    name = str(module)

        print(f"Moving {module.__module__}.{name} to GPU")

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
        mover = DiffModelMover.register(diff_model, lookahead_distance=8)
        mover.install()


def is_enabled(sd_model):
    return sd_model.lowvram
