import torch
from modules.devices import get_optimal_device

module_in_gpu = None
cpu = torch.device("cpu")
device = gpu = get_optimal_device()


def send_everything_to_cpu():
    global module_in_gpu

    if module_in_gpu is not None:
        module_in_gpu.to(cpu)

    module_in_gpu = None


def setup_for_low_vram(sd_model, use_medvram):
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

        module.to(gpu)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods
    def first_stage_model_encode_wrap(self, encoder, x):
        send_me_to_gpu(self, None)
        return encoder(x)

    def first_stage_model_decode_wrap(self, decoder, z):
        send_me_to_gpu(self, None)
        return decoder(z)

    # remove three big modules, cond, first_stage, and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = None, None, None
    sd_model.to(device)
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = stored

    # register hooks for those the first two models
    sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = lambda x, en=sd_model.first_stage_model.encode: first_stage_model_encode_wrap(sd_model.first_stage_model, en, x)
    sd_model.first_stage_model.decode = lambda z, de=sd_model.first_stage_model.decode: first_stage_model_decode_wrap(sd_model.first_stage_model, de, z)
    parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
