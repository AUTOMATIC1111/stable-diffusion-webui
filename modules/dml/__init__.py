import torch

def directml_init():
    from modules.dml.backend import DirectML # pylint: disable=ungrouped-imports
    from modules.dml.opts import override_opts # pylint: disable=ungrouped-imports
    # Alternative of torch.cuda for DirectML.
    torch.dml = DirectML

    override_opts()
