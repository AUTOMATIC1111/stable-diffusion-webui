### Impls of the SD3 core diffusion model and VAE

import torch
import math
import einops
from modules.models.sd3.mmdit import MMDiT
from PIL import Image


#################################################################################################
### MMDiT Model Wrapping
#################################################################################################


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""
    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""
    def __init__(self, shift=1.0, device=None, dtype=torch.float32, state_dict=None, prefix=""):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = state_dict[f"{prefix}x_embedder.proj.weight"].shape[2]
        depth = state_dict[f"{prefix}x_embedder.proj.weight"].shape[0] // 64
        num_patches = state_dict[f"{prefix}pos_embed"].shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = state_dict[f"{prefix}y_embedder.mlp.0.weight"].shape[1]
        context_shape = state_dict[f"{prefix}context_embedder.weight"].shape
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0]
            }
        }
        self.diffusion_model = MMDiT(input_size=None, pos_embed_scaling_factor=None, pos_embed_offset=None, pos_embed_max_size=pos_embed_max_size, patch_size=patch_size, in_channels=16, depth=depth, num_patches=num_patches, adm_in_channels=adm_in_channels, context_embedder_config=context_embedder_config, device=device, dtype=dtype)
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)

    def apply_model(self, x, sigma, c_crossattn=None, y=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype)).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timestep, cond, uncond, cond_scale):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(torch.cat([x, x]), torch.cat([timestep, timestep]), c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]), y=torch.cat([cond["y"], uncond["y"]]))
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""
    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor([
            [-0.0645,  0.0177,  0.1052], [ 0.0028,  0.0312,  0.0650],
            [ 0.1848,  0.0762,  0.0360], [ 0.0944,  0.0360,  0.0889],
            [ 0.0897,  0.0506, -0.0364], [-0.0020,  0.1203,  0.0284],
            [ 0.0855,  0.0118,  0.0283], [-0.0539,  0.0658,  0.1047],
            [-0.0057,  0.0116,  0.0700], [-0.0412,  0.0281, -0.0039],
            [ 0.1106,  0.1171,  0.1220], [-0.0248,  0.0682, -0.0481],
            [ 0.0815,  0.0846,  0.1207], [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259]
        ], device="cpu")
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


#################################################################################################
### K-Diffusion Sampling
#################################################################################################


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


#################################################################################################
### VAE
#################################################################################################


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = [einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous() for x in (q, k, v)]
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(self, ch=128, ch_mult=(1,2,4,4), num_res_blocks=2, in_channels=3, z_channels=16, dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, resolution=256, z_channels=16, dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)
