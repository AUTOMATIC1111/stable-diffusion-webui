from modules import shared

def override_opts():
    if shared.backend == shared.Backend.DIFFUSERS:
        shared.opts.diffusers_generator_device = "cpu" # DirectML does not support torch.Generator API.
