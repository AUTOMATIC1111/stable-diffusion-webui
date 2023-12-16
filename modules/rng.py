import torch

from modules import devices, rng_philox, shared


def randn(seed, shape, generator=None):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed."""

    manual_seed(seed)

    if shared.opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    return torch.randn(shape, device=devices.device, generator=generator)


def randn_local(seed, shape):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Does not change the global random number generator. You can only generate the seed's first tensor using this function."""

    if shared.opts.randn_source == "NV":
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=devices.device)

    local_device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


def randn_like(x):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    if shared.opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)

    if shared.opts.randn_source == "CPU" or x.device.type == 'mps':
        return torch.randn_like(x, device=devices.cpu).to(x.device)

    return torch.randn_like(x)


def randn_without_seed(shape, generator=None):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    if shared.opts.randn_source == "NV":
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    return torch.randn(shape, device=devices.device, generator=generator)


def manual_seed(seed):
    """Set up a global random number generator using the specified seed."""

    if shared.opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(seed):
    if shared.opts.randn_source == "NV":
        return rng_philox.Generator(seed)

    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class ImageRNG:
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(seed) for seed in seeds]

        self.is_first = True

    def first(self):
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(subseed, noise_shape)

            if noise_shape != self.shape:
                noise = randn(seed, noise_shape)
            else:
                noise = randn(seed, self.shape, generator=generator)

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty:ty + h, tx:tx + w] = noise[:, dy:dy + h, dx:dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]

        return torch.stack(xs).to(shared.device)

    def next(self):
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(self.shape, generator=generator)
            xs.append(x)

        return torch.stack(xs).to(shared.device)


devices.randn = randn
devices.randn_local = randn_local
devices.randn_like = randn_like
devices.randn_without_seed = randn_without_seed
devices.manual_seed = manual_seed
