from modules import shared

def override_opts():
    if shared.cmd_opts.backend.lower() == "diffusers":
        shared.opts.diffusers_generator_device = "cpu" # DirectML does not support torch.Generator API.
