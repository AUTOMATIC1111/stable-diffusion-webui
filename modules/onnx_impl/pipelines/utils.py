from typing import Union, List
import numpy as np
import torch


def extract_generator_seed(generator: Union[torch.Generator, List[torch.Generator]]) -> List[int]:
    if isinstance(generator, list):
        generator = [g.seed() for g in generator]
    else:
        generator = [generator.seed()]
    return generator


def randn_tensor(shape, dtype: np.dtype, generator: Union[torch.Generator, List[torch.Generator], int, List[int]]):
    if hasattr(generator, "seed") or (isinstance(generator, list) and hasattr(generator[0], "seed")):
        generator = extract_generator_seed(generator)
        if len(generator) == 1:
            generator = generator[0]
    return np.random.default_rng(generator).standard_normal(shape).astype(dtype)


def prepare_latents(
    init_noise_sigma: float,
    batch_size: int,
    height: int,
    width: int,
    dtype: np.dtype,
    generator: Union[torch.Generator, List[torch.Generator]],
    latents: Union[np.ndarray, None]=None,
    num_channels_latents=4,
    vae_scale_factor=8,
):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, dtype, generator)
    elif latents.shape != shape:
        raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * np.float64(init_noise_sigma)

    return latents
