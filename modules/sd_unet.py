import torch.nn
import ldm.modules.diffusionmodules.openaimodel

from modules import script_callbacks, shared, devices

unet_options = []
current_unet_option = None
current_unet = None


def list_unets():
    new_unets = script_callbacks.list_unets_callback()

    unet_options.clear()
    unet_options.extend(new_unets)


def get_unet_option(option=None):
    option = option or shared.opts.sd_unet

    if option == "None":
        return None

    if option == "Automatic":
        name = shared.sd_model.sd_checkpoint_info.model_name

        options = [x for x in unet_options if x.model_name == name]

        option = options[0].label if options else "None"

    return next(iter([x for x in unet_options if x.label == option]), None)


def apply_unet(option=None):
    global current_unet_option
    global current_unet

    new_option = get_unet_option(option)
    if new_option == current_unet_option:
        return

    if current_unet is not None:
        print(f"Dectivating unet: {current_unet.option.label}")
        current_unet.deactivate()

    current_unet_option = new_option
    if current_unet_option is None:
        current_unet = None

        if not shared.sd_model.lowvram:
            shared.sd_model.model.diffusion_model.to(devices.device)

        return

    shared.sd_model.model.diffusion_model.to(devices.cpu)
    devices.torch_gc()

    current_unet = current_unet_option.create_unet()
    current_unet.option = current_unet_option
    print(f"Activating unet: {current_unet.option.label}")
    current_unet.activate()


class SdUnetOption:
    model_name = None
    """name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this"""

    label = None
    """name of the unet in UI"""

    def create_unet(self):
        """returns SdUnet object to be used as a Unet instead of built-in unet when making pictures"""
        raise NotImplementedError()


class SdUnet(torch.nn.Module):
    def forward(self, x, timesteps, context, *args, **kwargs):
        raise NotImplementedError()

    def activate(self):
        pass

    def deactivate(self):
        pass


def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
    if current_unet is not None:
        return current_unet.forward(x, timesteps, context, *args, **kwargs)

    return ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui(self, x, timesteps, context, *args, **kwargs)

