import torch, math
from einops import rearrange, repeat
from .sd_unet import Timesteps, PushBlock, PopBlock, Attention, GEGLU, ResnetBlock, AttentionBlock, DownSampler, UpSampler


class TemporalResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        x = rearrange(hidden_states, "f c h w -> 1 c f h w")
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)
            emb = repeat(emb, "b c -> b c f 1 1", f=hidden_states.shape[0])
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        x = rearrange(x[0], "c f h w -> f c h w")
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TemporalTimesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TemporalAttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, cross_attention_dim=None):
        super().__init__()

        self.positional_embedding = TemporalTimesteps(in_channels, True, 0)
        self.positional_embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(in_channels * 4, in_channels)
        )

        self.norm_in = torch.nn.LayerNorm(in_channels)
        self.act_fn_in = GEGLU(in_channels, in_channels * 4)
        self.ff_in = torch.nn.Linear(in_channels * 4, in_channels)

        self.norm1 = torch.nn.LayerNorm(in_channels)
        self.attn1 = Attention(
            q_dim=in_channels,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True
        )

        self.norm2 = torch.nn.LayerNorm(in_channels)
        self.attn2 = Attention(
            q_dim=in_channels,
            kv_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias_out=True
        )

        self.norm_out = torch.nn.LayerNorm(in_channels)
        self.act_fn_out = GEGLU(in_channels, in_channels * 4)
        self.ff_out = torch.nn.Linear(in_channels * 4, in_channels)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):

        batch, inner_dim, height, width = hidden_states.shape
        pos_emb = torch.arange(batch)
        pos_emb = self.positional_embedding(pos_emb).to(dtype=hidden_states.dtype, device=hidden_states.device)
        pos_emb = self.positional_embedding_proj(pos_emb)[None, :, :]

        hidden_states = hidden_states.permute(2, 3, 0, 1).reshape(height * width, batch, inner_dim)
        hidden_states = hidden_states + pos_emb

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.act_fn_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=text_emb.repeat(height * width, 1))
        hidden_states = attn_output + hidden_states

        residual = hidden_states
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.act_fn_out(hidden_states)
        hidden_states = self.ff_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states.reshape(height, width, batch, inner_dim).permute(2, 3, 0, 1)

        return hidden_states, time_emb, text_emb, res_stack
    

class PopMixBlock(torch.nn.Module):
    def __init__(self, in_channels=None):
        super().__init__()
        self.mix_factor = torch.nn.Parameter(torch.Tensor([0.5]))
        self.need_proj = in_channels is not None
        if self.need_proj:
            self.proj = torch.nn.Linear(in_channels, in_channels)
    
    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        res_hidden_states = res_stack.pop()
        alpha = torch.sigmoid(self.mix_factor)
        hidden_states = alpha * res_hidden_states + (1 - alpha) * hidden_states
        if self.need_proj:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.proj(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            res_hidden_states = res_stack.pop()
            hidden_states = hidden_states + res_hidden_states
        return hidden_states, time_emb, text_emb, res_stack


class SVDUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_proj = Timesteps(320)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.add_time_proj = Timesteps(256)
        self.add_time_embedding = torch.nn.Sequential(
            torch.nn.Linear(768, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.conv_in = torch.nn.Conv2d(8, 320, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(320, 320, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024),        PopMixBlock(320),  PushBlock(),
            ResnetBlock(320, 320, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024),        PopMixBlock(320),  PushBlock(),
            DownSampler(320), PushBlock(),
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(320, 640, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024),       PopMixBlock(640),  PushBlock(),
            ResnetBlock(640, 640, 1280, eps=1e-6),                      PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024),       PopMixBlock(640),  PushBlock(),
            DownSampler(640), PushBlock(),
            # CrossAttnDownBlockSpatioTemporal
            ResnetBlock(640, 1280, 1280, eps=1e-6),                     PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280), PushBlock(),
            ResnetBlock(1280, 1280, 1280, eps=1e-6),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280), PushBlock(),
            DownSampler(1280), PushBlock(),
            # DownBlockSpatioTemporal
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),     PushBlock(),
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),     PushBlock(),
            # UNetMidBlockSpatioTemporal
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280),
            ResnetBlock(1280, 1280, 1280, eps=1e-5),                    PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            # UpBlockSpatioTemporal
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            PopBlock(), ResnetBlock(2560, 1280, 1280, eps=1e-6),        PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-5), PopMixBlock(),
            UpSampler(1280),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(2560, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280),
            PopBlock(),        ResnetBlock(2560, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280),
            PopBlock(),        ResnetBlock(1920, 1280, 1280, eps=1e-6), PushBlock(), TemporalResnetBlock(1280, 1280, 1280, eps=1e-6), PopMixBlock(),     PushBlock(),
            AttentionBlock(20, 64, 1280, 1, 1024, need_proj_out=False), PushBlock(), TemporalAttentionBlock(20, 64, 1280, 1024),      PopMixBlock(1280),
            UpSampler(1280),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(1920, 640, 1280, eps=1e-6),  PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024),       PopMixBlock(640),
            PopBlock(),        ResnetBlock(1280, 640, 1280, eps=1e-6),  PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024),       PopMixBlock(640),
            PopBlock(),        ResnetBlock(960, 640, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(640, 640, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(10, 64, 640, 1, 1024, need_proj_out=False),  PushBlock(), TemporalAttentionBlock(10, 64, 640, 1024),       PopMixBlock(640),
            UpSampler(640),
            # CrossAttnUpBlockSpatioTemporal
            PopBlock(),        ResnetBlock(960, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024),        PopMixBlock(320),
            PopBlock(),        ResnetBlock(640, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024),        PopMixBlock(320),
            PopBlock(),        ResnetBlock(640, 320, 1280, eps=1e-6),   PushBlock(), TemporalResnetBlock(320, 320, 1280, eps=1e-6),   PopMixBlock(),     PushBlock(),
            AttentionBlock(5, 64, 320, 1, 1024, need_proj_out=False),   PushBlock(), TemporalAttentionBlock(5, 64, 320, 1024),        PopMixBlock(320),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(32, 320, eps=1e-05, affine=True)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, sample, timestep, encoder_hidden_states, add_time_id, **kwargs):
        # 1. time
        t_emb = self.time_proj(timestep[None]).to(sample.dtype)
        t_emb = self.time_embedding(t_emb)

        add_embeds = self.add_time_proj(add_time_id.flatten()).to(sample.dtype)
        add_embeds = add_embeds.reshape((-1, 768))
        add_embeds = self.add_time_embedding(add_embeds)

        time_emb = t_emb + add_embeds

        # 2. pre-process
        height, width = sample.shape[2], sample.shape[3]
        hidden_states = self.conv_in(sample)
        text_emb = encoder_hidden_states
        res_stack = [hidden_states]

        # 3. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        # 4. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states
    
    def state_dict_converter(self):
        return SVDUNetStateDictConverter()
    


class SVDUNetStateDictConverter:
    def __init__(self):
        pass

    def get_block_name(self, names):
        if names[0] in ["down_blocks", "mid_block", "up_blocks"]:
            if names[4] in ["norm", "proj_in"]:
                return ".".join(names[:4] + ["transformer_blocks"])
            elif names[4] in ["time_pos_embed"]:
                return ".".join(names[:4] + ["temporal_transformer_blocks"])
            elif names[4] in ["proj_out"]:
                return ".".join(names[:4] + ["time_mixer"])
            else:
                return ".".join(names[:5])
        return ""

    def from_diffusers(self, state_dict):
        rename_dict = {
            "time_embedding.linear_1": "time_embedding.0",
            "time_embedding.linear_2": "time_embedding.2",
            "add_embedding.linear_1": "add_time_embedding.0",
            "add_embedding.linear_2": "add_time_embedding.2",
            "conv_in": "conv_in",
            "conv_norm_out": "conv_norm_out",
            "conv_out": "conv_out",
        }
        blocks_rename_dict = [
            "down_blocks.0.resnets.0.spatial_res_block", None, "down_blocks.0.resnets.0.temporal_res_block", "down_blocks.0.resnets.0.time_mixer", None,
            "down_blocks.0.attentions.0.transformer_blocks", None, "down_blocks.0.attentions.0.temporal_transformer_blocks", "down_blocks.0.attentions.0.time_mixer", None,
            "down_blocks.0.resnets.1.spatial_res_block", None, "down_blocks.0.resnets.1.temporal_res_block", "down_blocks.0.resnets.1.time_mixer", None,
            "down_blocks.0.attentions.1.transformer_blocks", None, "down_blocks.0.attentions.1.temporal_transformer_blocks", "down_blocks.0.attentions.1.time_mixer", None,
            "down_blocks.0.downsamplers.0.conv", None,
            "down_blocks.1.resnets.0.spatial_res_block", None, "down_blocks.1.resnets.0.temporal_res_block", "down_blocks.1.resnets.0.time_mixer", None,
            "down_blocks.1.attentions.0.transformer_blocks", None, "down_blocks.1.attentions.0.temporal_transformer_blocks", "down_blocks.1.attentions.0.time_mixer", None,
            "down_blocks.1.resnets.1.spatial_res_block", None, "down_blocks.1.resnets.1.temporal_res_block", "down_blocks.1.resnets.1.time_mixer", None,
            "down_blocks.1.attentions.1.transformer_blocks", None, "down_blocks.1.attentions.1.temporal_transformer_blocks", "down_blocks.1.attentions.1.time_mixer", None,
            "down_blocks.1.downsamplers.0.conv", None,
            "down_blocks.2.resnets.0.spatial_res_block", None, "down_blocks.2.resnets.0.temporal_res_block", "down_blocks.2.resnets.0.time_mixer", None,
            "down_blocks.2.attentions.0.transformer_blocks", None, "down_blocks.2.attentions.0.temporal_transformer_blocks", "down_blocks.2.attentions.0.time_mixer", None,
            "down_blocks.2.resnets.1.spatial_res_block", None, "down_blocks.2.resnets.1.temporal_res_block", "down_blocks.2.resnets.1.time_mixer", None,
            "down_blocks.2.attentions.1.transformer_blocks", None, "down_blocks.2.attentions.1.temporal_transformer_blocks", "down_blocks.2.attentions.1.time_mixer", None,
            "down_blocks.2.downsamplers.0.conv", None,
            "down_blocks.3.resnets.0.spatial_res_block", None, "down_blocks.3.resnets.0.temporal_res_block", "down_blocks.3.resnets.0.time_mixer", None,
            "down_blocks.3.resnets.1.spatial_res_block", None, "down_blocks.3.resnets.1.temporal_res_block", "down_blocks.3.resnets.1.time_mixer", None,
            "mid_block.mid_block.resnets.0.spatial_res_block", None, "mid_block.mid_block.resnets.0.temporal_res_block", "mid_block.mid_block.resnets.0.time_mixer", None,
            "mid_block.mid_block.attentions.0.transformer_blocks", None, "mid_block.mid_block.attentions.0.temporal_transformer_blocks", "mid_block.mid_block.attentions.0.time_mixer",
            "mid_block.mid_block.resnets.1.spatial_res_block", None, "mid_block.mid_block.resnets.1.temporal_res_block", "mid_block.mid_block.resnets.1.time_mixer",
            None, "up_blocks.0.resnets.0.spatial_res_block", None, "up_blocks.0.resnets.0.temporal_res_block", "up_blocks.0.resnets.0.time_mixer",
            None, "up_blocks.0.resnets.1.spatial_res_block", None, "up_blocks.0.resnets.1.temporal_res_block", "up_blocks.0.resnets.1.time_mixer",
            None, "up_blocks.0.resnets.2.spatial_res_block", None, "up_blocks.0.resnets.2.temporal_res_block", "up_blocks.0.resnets.2.time_mixer",
            "up_blocks.0.upsamplers.0.conv",
            None, "up_blocks.1.resnets.0.spatial_res_block", None, "up_blocks.1.resnets.0.temporal_res_block", "up_blocks.1.resnets.0.time_mixer", None,
            "up_blocks.1.attentions.0.transformer_blocks", None, "up_blocks.1.attentions.0.temporal_transformer_blocks", "up_blocks.1.attentions.0.time_mixer",
            None, "up_blocks.1.resnets.1.spatial_res_block", None, "up_blocks.1.resnets.1.temporal_res_block", "up_blocks.1.resnets.1.time_mixer", None,
            "up_blocks.1.attentions.1.transformer_blocks", None, "up_blocks.1.attentions.1.temporal_transformer_blocks", "up_blocks.1.attentions.1.time_mixer",
            None, "up_blocks.1.resnets.2.spatial_res_block", None, "up_blocks.1.resnets.2.temporal_res_block", "up_blocks.1.resnets.2.time_mixer", None,
            "up_blocks.1.attentions.2.transformer_blocks", None, "up_blocks.1.attentions.2.temporal_transformer_blocks", "up_blocks.1.attentions.2.time_mixer",
            "up_blocks.1.upsamplers.0.conv",
            None, "up_blocks.2.resnets.0.spatial_res_block", None, "up_blocks.2.resnets.0.temporal_res_block", "up_blocks.2.resnets.0.time_mixer", None,
            "up_blocks.2.attentions.0.transformer_blocks", None, "up_blocks.2.attentions.0.temporal_transformer_blocks", "up_blocks.2.attentions.0.time_mixer",
            None, "up_blocks.2.resnets.1.spatial_res_block", None, "up_blocks.2.resnets.1.temporal_res_block", "up_blocks.2.resnets.1.time_mixer", None,
            "up_blocks.2.attentions.1.transformer_blocks", None, "up_blocks.2.attentions.1.temporal_transformer_blocks", "up_blocks.2.attentions.1.time_mixer",
            None, "up_blocks.2.resnets.2.spatial_res_block", None, "up_blocks.2.resnets.2.temporal_res_block", "up_blocks.2.resnets.2.time_mixer", None,
            "up_blocks.2.attentions.2.transformer_blocks", None, "up_blocks.2.attentions.2.temporal_transformer_blocks", "up_blocks.2.attentions.2.time_mixer",
            "up_blocks.2.upsamplers.0.conv",
            None, "up_blocks.3.resnets.0.spatial_res_block", None, "up_blocks.3.resnets.0.temporal_res_block", "up_blocks.3.resnets.0.time_mixer", None,
            "up_blocks.3.attentions.0.transformer_blocks", None, "up_blocks.3.attentions.0.temporal_transformer_blocks", "up_blocks.3.attentions.0.time_mixer",
            None, "up_blocks.3.resnets.1.spatial_res_block", None, "up_blocks.3.resnets.1.temporal_res_block", "up_blocks.3.resnets.1.time_mixer", None,
            "up_blocks.3.attentions.1.transformer_blocks", None, "up_blocks.3.attentions.1.temporal_transformer_blocks", "up_blocks.3.attentions.1.time_mixer",
            None, "up_blocks.3.resnets.2.spatial_res_block", None, "up_blocks.3.resnets.2.temporal_res_block", "up_blocks.3.resnets.2.time_mixer", None,
            "up_blocks.3.attentions.2.transformer_blocks", None, "up_blocks.3.attentions.2.temporal_transformer_blocks", "up_blocks.3.attentions.2.time_mixer",
        ]
        blocks_rename_dict = {i:j for j,i in enumerate(blocks_rename_dict) if i is not None}
        state_dict_ = {}
        for name, param in sorted(state_dict.items()):
            names = name.split(".")
            if names[0] == "mid_block":
                names = ["mid_block"] + names
            if names[-1] in ["weight", "bias"]:
                name_prefix = ".".join(names[:-1])
                if name_prefix in rename_dict:
                    state_dict_[rename_dict[name_prefix] + "." + names[-1]] = param
                else:
                    block_name = self.get_block_name(names)
                    if "resnets" in block_name and block_name in blocks_rename_dict:
                        rename = ".".join(["blocks", str(blocks_rename_dict[block_name])] + names[5:])
                        state_dict_[rename] = param
                    elif ("downsamplers" in block_name or "upsamplers" in block_name) and block_name in blocks_rename_dict:
                        rename = ".".join(["blocks", str(blocks_rename_dict[block_name])] + names[-2:])
                        state_dict_[rename] = param
                    elif "attentions" in block_name and block_name in blocks_rename_dict:
                        attention_id = names[5]
                        if "transformer_blocks" in names:
                            suffix_dict = {
                                "attn1.to_out.0": "attn1.to_out",
                                "attn2.to_out.0": "attn2.to_out",
                                "ff.net.0.proj": "act_fn.proj",
                                "ff.net.2": "ff",
                            }
                            suffix = ".".join(names[6:-1])
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), "transformer_blocks", attention_id, suffix, names[-1]])
                        elif "temporal_transformer_blocks" in names:
                            suffix_dict = {
                                "attn1.to_out.0": "attn1.to_out",
                                "attn2.to_out.0": "attn2.to_out",
                                "ff_in.net.0.proj": "act_fn_in.proj",
                                "ff_in.net.2": "ff_in",
                                "ff.net.0.proj": "act_fn_out.proj",
                                "ff.net.2": "ff_out",
                                "norm3": "norm_out",
                            }
                            suffix = ".".join(names[6:-1])
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), suffix, names[-1]])
                        elif "time_mixer" in block_name:
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), "proj", names[-1]])
                        else:
                            suffix_dict = {
                                "linear_1": "positional_embedding_proj.0",
                                "linear_2": "positional_embedding_proj.2",
                            }
                            suffix = names[-2]
                            suffix = suffix_dict.get(suffix, suffix)
                            rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), suffix, names[-1]])
                        state_dict_[rename] = param
                    else:
                        print(name)
            else:
                block_name = self.get_block_name(names)
                if len(block_name)>0 and block_name in blocks_rename_dict:
                    rename = ".".join(["blocks", str(blocks_rename_dict[block_name]), names[-1]])
                    state_dict_[rename] = param
        return state_dict_
