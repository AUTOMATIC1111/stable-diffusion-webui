import torch

# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
from modules import errors

has_mps = getattr(torch, 'has_mps', False)

cpu = torch.device("cpu")


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if has_mps:
        return torch.device("mps")

    return cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")
