import torch, math
from .attention import Attention
from .tiler import Tiler


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        timesteps = timesteps.unsqueeze(-1)
        emb = timesteps.float() * torch.exp(exponent)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb


class GEGLU(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.gelu(gate)


class BasicTransformerBlock(torch.nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim):
        super().__init__()

        # 1. Self-Attn
        self.norm1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = Attention(q_dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 2. Cross-Attn
        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = Attention(q_dim=dim, kv_dim=cross_attention_dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 3. Feed-forward
        self.norm3 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.act_fn = GEGLU(dim, dim * 4)
        self.ff = torch.nn.Linear(dim * 4, dim)


    def forward(self, hidden_states, encoder_hidden_states):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None,)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.act_fn(norm_hidden_states)
        ff_output = self.ff(ff_output)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DownSampler(torch.nn.Module):
    def __init__(self, channels, padding=1, extra_padding=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        self.extra_padding = extra_padding

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        if self.extra_padding:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        if time_emb is not None:
            emb = self.nonlinearity(time_emb)
            emb = self.time_emb_proj(emb)[:, :, None, None]
            x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class AttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, cross_attention_dim=None, norm_num_groups=32, eps=1e-5, need_proj_out=True):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim
            )
            for d in range(num_layers)
        ])
        self.need_proj_out = need_proj_out
        if need_proj_out:
            self.proj_out = torch.nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, cross_frame_attention=False, **kwargs):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        if cross_frame_attention:
            hidden_states = hidden_states.reshape(1, batch * height * width, inner_dim)
            encoder_hidden_states = text_emb.mean(dim=0, keepdim=True)
        else:
            encoder_hidden_states = text_emb
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states
            )
        if cross_frame_attention:
            hidden_states = hidden_states.reshape(batch, height * width, inner_dim)

        if self.need_proj_out:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        return hidden_states, time_emb, text_emb, res_stack


class PushBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_stack.append(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class PopBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack, **kwargs):
        res_hidden_states = res_stack.pop()
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        return hidden_states, time_emb, text_emb, res_stack


class SDUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_proj = Timesteps(320)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.conv_in = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # CrossAttnDownBlock2D
            ResnetBlock(320, 320, 1280),
            AttentionBlock(8, 40, 320, 1, 768, eps=1e-6),
            PushBlock(),
            ResnetBlock(320, 320, 1280),
            AttentionBlock(8, 40, 320, 1, 768, eps=1e-6),
            PushBlock(),
            DownSampler(320),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(320, 640, 1280),
            AttentionBlock(8, 80, 640, 1, 768, eps=1e-6),
            PushBlock(),
            ResnetBlock(640, 640, 1280),
            AttentionBlock(8, 80, 640, 1, 768, eps=1e-6),
            PushBlock(),
            DownSampler(640),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(640, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            PushBlock(),
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            PushBlock(),
            DownSampler(1280),
            PushBlock(),
            # DownBlock2D
            ResnetBlock(1280, 1280, 1280),
            PushBlock(),
            ResnetBlock(1280, 1280, 1280),
            PushBlock(),
            # UNetMidBlock2DCrossAttn
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            ResnetBlock(1280, 1280, 1280),
            # UpBlock2D
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            UpSampler(1280),
            # CrossAttnUpBlock2D
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(1920, 1280, 1280),
            AttentionBlock(8, 160, 1280, 1, 768, eps=1e-6),
            UpSampler(1280),
            # CrossAttnUpBlock2D
            PopBlock(),
            ResnetBlock(1920, 640, 1280),
            AttentionBlock(8, 80, 640, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(1280, 640, 1280),
            AttentionBlock(8, 80, 640, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(960, 640, 1280),
            AttentionBlock(8, 80, 640, 1, 768, eps=1e-6),
            UpSampler(640),
            # CrossAttnUpBlock2D
            PopBlock(),
            ResnetBlock(960, 320, 1280),
            AttentionBlock(8, 40, 320, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(640, 320, 1280),
            AttentionBlock(8, 40, 320, 1, 768, eps=1e-6),
            PopBlock(),
            ResnetBlock(640, 320, 1280),
            AttentionBlock(8, 40, 320, 1, 768, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(320, 4, kernel_size=3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # 1. time
        time_emb = self.time_proj(timestep[None]).to(sample.dtype)
        time_emb = self.time_embedding(time_emb)

        # 2. pre-process
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
        return SDUNetStateDictConverter()


class SDUNetStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # architecture
        block_types = [
            'ResnetBlock', 'AttentionBlock', 'PushBlock', 'ResnetBlock', 'AttentionBlock', 'PushBlock', 'DownSampler', 'PushBlock',
            'ResnetBlock', 'AttentionBlock', 'PushBlock', 'ResnetBlock', 'AttentionBlock', 'PushBlock', 'DownSampler', 'PushBlock',
            'ResnetBlock', 'AttentionBlock', 'PushBlock', 'ResnetBlock', 'AttentionBlock', 'PushBlock', 'DownSampler', 'PushBlock',
            'ResnetBlock', 'PushBlock', 'ResnetBlock', 'PushBlock', 
            'ResnetBlock', 'AttentionBlock', 'ResnetBlock',
            'PopBlock', 'ResnetBlock', 'PopBlock', 'ResnetBlock', 'PopBlock', 'ResnetBlock', 'UpSampler',
            'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock', 'UpSampler',
            'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock', 'UpSampler',
            'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock', 'PopBlock', 'ResnetBlock', 'AttentionBlock'
        ]

        # Rename each parameter
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "AttentionBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "AttentionBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            if names[0] in ["conv_in", "conv_norm_out", "conv_out"]:
                pass
            elif names[0] in ["time_embedding", "add_embedding"]:
                if names[0] == "add_embedding":
                    names[0] = "add_time_embedding"
                names[1] = {"linear_1": "0", "linear_2": "2"}[names[1]]
            elif names[0] in ["down_blocks", "mid_block", "up_blocks"]:
                if names[0] == "mid_block":
                    names.insert(1, "0")
                block_type = {"resnets": "ResnetBlock", "attentions": "AttentionBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[2]]
                block_type_with_id = ".".join(names[:4])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:4])
                names = ["blocks", str(block_id[block_type])] + names[4:]
                if "ff" in names:
                    ff_index = names.index("ff")
                    component = ".".join(names[ff_index:ff_index+3])
                    component = {"ff.net.0": "act_fn", "ff.net.2": "ff"}[component]
                    names = names[:ff_index] + [component] + names[ff_index+3:]
                if "to_out" in names:
                    names.pop(names.index("to_out") + 1)
            else:
                raise ValueError(f"Unknown parameters: {name}")
            rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if ".proj_in." in name or ".proj_out." in name:
                param = param.squeeze()
            state_dict_[rename_dict[name]] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        rename_dict = {
            "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
            "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
            "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias": "blocks.0.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": "blocks.0.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.bias": "blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.1.0.in_layers.0.weight": "blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.1.0.in_layers.2.bias": "blocks.0.conv1.bias",
            "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": "blocks.0.conv1.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.0.bias": "blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.1.0.out_layers.0.weight": "blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.1.0.out_layers.3.bias": "blocks.0.conv2.bias",
            "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": "blocks.0.conv2.weight",
            "model.diffusion_model.input_blocks.1.1.norm.bias": "blocks.1.norm.bias",
            "model.diffusion_model.input_blocks.1.1.norm.weight": "blocks.1.norm.weight",
            "model.diffusion_model.input_blocks.1.1.proj_in.bias": "blocks.1.proj_in.bias",
            "model.diffusion_model.input_blocks.1.1.proj_in.weight": "blocks.1.proj_in.weight",
            "model.diffusion_model.input_blocks.1.1.proj_out.bias": "blocks.1.proj_out.bias",
            "model.diffusion_model.input_blocks.1.1.proj_out.weight": "blocks.1.proj_out.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": "blocks.1.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.1.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.1.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": "blocks.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": "blocks.1.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": "blocks.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.1.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.1.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": "blocks.1.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": "blocks.1.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.1.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.1.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias": "blocks.1.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight": "blocks.1.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias": "blocks.1.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight": "blocks.1.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias": "blocks.1.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight": "blocks.1.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias": "blocks.1.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight": "blocks.1.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.10.0.emb_layers.1.bias": "blocks.24.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.10.0.emb_layers.1.weight": "blocks.24.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.10.0.in_layers.0.bias": "blocks.24.norm1.bias",
            "model.diffusion_model.input_blocks.10.0.in_layers.0.weight": "blocks.24.norm1.weight",
            "model.diffusion_model.input_blocks.10.0.in_layers.2.bias": "blocks.24.conv1.bias",
            "model.diffusion_model.input_blocks.10.0.in_layers.2.weight": "blocks.24.conv1.weight",
            "model.diffusion_model.input_blocks.10.0.out_layers.0.bias": "blocks.24.norm2.bias",
            "model.diffusion_model.input_blocks.10.0.out_layers.0.weight": "blocks.24.norm2.weight",
            "model.diffusion_model.input_blocks.10.0.out_layers.3.bias": "blocks.24.conv2.bias",
            "model.diffusion_model.input_blocks.10.0.out_layers.3.weight": "blocks.24.conv2.weight",
            "model.diffusion_model.input_blocks.11.0.emb_layers.1.bias": "blocks.26.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.11.0.emb_layers.1.weight": "blocks.26.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.11.0.in_layers.0.bias": "blocks.26.norm1.bias",
            "model.diffusion_model.input_blocks.11.0.in_layers.0.weight": "blocks.26.norm1.weight",
            "model.diffusion_model.input_blocks.11.0.in_layers.2.bias": "blocks.26.conv1.bias",
            "model.diffusion_model.input_blocks.11.0.in_layers.2.weight": "blocks.26.conv1.weight",
            "model.diffusion_model.input_blocks.11.0.out_layers.0.bias": "blocks.26.norm2.bias",
            "model.diffusion_model.input_blocks.11.0.out_layers.0.weight": "blocks.26.norm2.weight",
            "model.diffusion_model.input_blocks.11.0.out_layers.3.bias": "blocks.26.conv2.bias",
            "model.diffusion_model.input_blocks.11.0.out_layers.3.weight": "blocks.26.conv2.weight",
            "model.diffusion_model.input_blocks.2.0.emb_layers.1.bias": "blocks.3.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.2.0.emb_layers.1.weight": "blocks.3.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.2.0.in_layers.0.bias": "blocks.3.norm1.bias",
            "model.diffusion_model.input_blocks.2.0.in_layers.0.weight": "blocks.3.norm1.weight",
            "model.diffusion_model.input_blocks.2.0.in_layers.2.bias": "blocks.3.conv1.bias",
            "model.diffusion_model.input_blocks.2.0.in_layers.2.weight": "blocks.3.conv1.weight",
            "model.diffusion_model.input_blocks.2.0.out_layers.0.bias": "blocks.3.norm2.bias",
            "model.diffusion_model.input_blocks.2.0.out_layers.0.weight": "blocks.3.norm2.weight",
            "model.diffusion_model.input_blocks.2.0.out_layers.3.bias": "blocks.3.conv2.bias",
            "model.diffusion_model.input_blocks.2.0.out_layers.3.weight": "blocks.3.conv2.weight",
            "model.diffusion_model.input_blocks.2.1.norm.bias": "blocks.4.norm.bias",
            "model.diffusion_model.input_blocks.2.1.norm.weight": "blocks.4.norm.weight",
            "model.diffusion_model.input_blocks.2.1.proj_in.bias": "blocks.4.proj_in.bias",
            "model.diffusion_model.input_blocks.2.1.proj_in.weight": "blocks.4.proj_in.weight",
            "model.diffusion_model.input_blocks.2.1.proj_out.bias": "blocks.4.proj_out.bias",
            "model.diffusion_model.input_blocks.2.1.proj_out.weight": "blocks.4.proj_out.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight": "blocks.4.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.4.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.4.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight": "blocks.4.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight": "blocks.4.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": "blocks.4.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.4.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.4.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight": "blocks.4.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": "blocks.4.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.4.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.4.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias": "blocks.4.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight": "blocks.4.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias": "blocks.4.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight": "blocks.4.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias": "blocks.4.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight": "blocks.4.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias": "blocks.4.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight": "blocks.4.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.3.0.op.bias": "blocks.6.conv.bias",
            "model.diffusion_model.input_blocks.3.0.op.weight": "blocks.6.conv.weight",
            "model.diffusion_model.input_blocks.4.0.emb_layers.1.bias": "blocks.8.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.4.0.emb_layers.1.weight": "blocks.8.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.4.0.in_layers.0.bias": "blocks.8.norm1.bias",
            "model.diffusion_model.input_blocks.4.0.in_layers.0.weight": "blocks.8.norm1.weight",
            "model.diffusion_model.input_blocks.4.0.in_layers.2.bias": "blocks.8.conv1.bias",
            "model.diffusion_model.input_blocks.4.0.in_layers.2.weight": "blocks.8.conv1.weight",
            "model.diffusion_model.input_blocks.4.0.out_layers.0.bias": "blocks.8.norm2.bias",
            "model.diffusion_model.input_blocks.4.0.out_layers.0.weight": "blocks.8.norm2.weight",
            "model.diffusion_model.input_blocks.4.0.out_layers.3.bias": "blocks.8.conv2.bias",
            "model.diffusion_model.input_blocks.4.0.out_layers.3.weight": "blocks.8.conv2.weight",
            "model.diffusion_model.input_blocks.4.0.skip_connection.bias": "blocks.8.conv_shortcut.bias",
            "model.diffusion_model.input_blocks.4.0.skip_connection.weight": "blocks.8.conv_shortcut.weight",
            "model.diffusion_model.input_blocks.4.1.norm.bias": "blocks.9.norm.bias",
            "model.diffusion_model.input_blocks.4.1.norm.weight": "blocks.9.norm.weight",
            "model.diffusion_model.input_blocks.4.1.proj_in.bias": "blocks.9.proj_in.bias",
            "model.diffusion_model.input_blocks.4.1.proj_in.weight": "blocks.9.proj_in.weight",
            "model.diffusion_model.input_blocks.4.1.proj_out.bias": "blocks.9.proj_out.bias",
            "model.diffusion_model.input_blocks.4.1.proj_out.weight": "blocks.9.proj_out.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "blocks.9.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.9.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.9.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "blocks.9.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "blocks.9.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "blocks.9.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.9.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.9.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "blocks.9.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "blocks.9.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.9.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.9.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "blocks.9.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "blocks.9.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias": "blocks.9.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight": "blocks.9.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias": "blocks.9.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight": "blocks.9.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias": "blocks.9.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight": "blocks.9.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.5.0.emb_layers.1.bias": "blocks.11.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.5.0.emb_layers.1.weight": "blocks.11.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.5.0.in_layers.0.bias": "blocks.11.norm1.bias",
            "model.diffusion_model.input_blocks.5.0.in_layers.0.weight": "blocks.11.norm1.weight",
            "model.diffusion_model.input_blocks.5.0.in_layers.2.bias": "blocks.11.conv1.bias",
            "model.diffusion_model.input_blocks.5.0.in_layers.2.weight": "blocks.11.conv1.weight",
            "model.diffusion_model.input_blocks.5.0.out_layers.0.bias": "blocks.11.norm2.bias",
            "model.diffusion_model.input_blocks.5.0.out_layers.0.weight": "blocks.11.norm2.weight",
            "model.diffusion_model.input_blocks.5.0.out_layers.3.bias": "blocks.11.conv2.bias",
            "model.diffusion_model.input_blocks.5.0.out_layers.3.weight": "blocks.11.conv2.weight",
            "model.diffusion_model.input_blocks.5.1.norm.bias": "blocks.12.norm.bias",
            "model.diffusion_model.input_blocks.5.1.norm.weight": "blocks.12.norm.weight",
            "model.diffusion_model.input_blocks.5.1.proj_in.bias": "blocks.12.proj_in.bias",
            "model.diffusion_model.input_blocks.5.1.proj_in.weight": "blocks.12.proj_in.weight",
            "model.diffusion_model.input_blocks.5.1.proj_out.bias": "blocks.12.proj_out.bias",
            "model.diffusion_model.input_blocks.5.1.proj_out.weight": "blocks.12.proj_out.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "blocks.12.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.12.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.12.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "blocks.12.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "blocks.12.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "blocks.12.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.12.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.12.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "blocks.12.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "blocks.12.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.12.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.12.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "blocks.12.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "blocks.12.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias": "blocks.12.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight": "blocks.12.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias": "blocks.12.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight": "blocks.12.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias": "blocks.12.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight": "blocks.12.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.6.0.op.bias": "blocks.14.conv.bias",
            "model.diffusion_model.input_blocks.6.0.op.weight": "blocks.14.conv.weight",
            "model.diffusion_model.input_blocks.7.0.emb_layers.1.bias": "blocks.16.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.7.0.emb_layers.1.weight": "blocks.16.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.7.0.in_layers.0.bias": "blocks.16.norm1.bias",
            "model.diffusion_model.input_blocks.7.0.in_layers.0.weight": "blocks.16.norm1.weight",
            "model.diffusion_model.input_blocks.7.0.in_layers.2.bias": "blocks.16.conv1.bias",
            "model.diffusion_model.input_blocks.7.0.in_layers.2.weight": "blocks.16.conv1.weight",
            "model.diffusion_model.input_blocks.7.0.out_layers.0.bias": "blocks.16.norm2.bias",
            "model.diffusion_model.input_blocks.7.0.out_layers.0.weight": "blocks.16.norm2.weight",
            "model.diffusion_model.input_blocks.7.0.out_layers.3.bias": "blocks.16.conv2.bias",
            "model.diffusion_model.input_blocks.7.0.out_layers.3.weight": "blocks.16.conv2.weight",
            "model.diffusion_model.input_blocks.7.0.skip_connection.bias": "blocks.16.conv_shortcut.bias",
            "model.diffusion_model.input_blocks.7.0.skip_connection.weight": "blocks.16.conv_shortcut.weight",
            "model.diffusion_model.input_blocks.7.1.norm.bias": "blocks.17.norm.bias",
            "model.diffusion_model.input_blocks.7.1.norm.weight": "blocks.17.norm.weight",
            "model.diffusion_model.input_blocks.7.1.proj_in.bias": "blocks.17.proj_in.bias",
            "model.diffusion_model.input_blocks.7.1.proj_in.weight": "blocks.17.proj_in.weight",
            "model.diffusion_model.input_blocks.7.1.proj_out.bias": "blocks.17.proj_out.bias",
            "model.diffusion_model.input_blocks.7.1.proj_out.weight": "blocks.17.proj_out.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "blocks.17.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.17.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.17.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "blocks.17.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "blocks.17.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "blocks.17.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.17.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.17.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "blocks.17.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "blocks.17.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.17.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.17.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "blocks.17.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "blocks.17.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias": "blocks.17.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight": "blocks.17.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias": "blocks.17.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight": "blocks.17.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias": "blocks.17.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight": "blocks.17.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.8.0.emb_layers.1.bias": "blocks.19.time_emb_proj.bias",
            "model.diffusion_model.input_blocks.8.0.emb_layers.1.weight": "blocks.19.time_emb_proj.weight",
            "model.diffusion_model.input_blocks.8.0.in_layers.0.bias": "blocks.19.norm1.bias",
            "model.diffusion_model.input_blocks.8.0.in_layers.0.weight": "blocks.19.norm1.weight",
            "model.diffusion_model.input_blocks.8.0.in_layers.2.bias": "blocks.19.conv1.bias",
            "model.diffusion_model.input_blocks.8.0.in_layers.2.weight": "blocks.19.conv1.weight",
            "model.diffusion_model.input_blocks.8.0.out_layers.0.bias": "blocks.19.norm2.bias",
            "model.diffusion_model.input_blocks.8.0.out_layers.0.weight": "blocks.19.norm2.weight",
            "model.diffusion_model.input_blocks.8.0.out_layers.3.bias": "blocks.19.conv2.bias",
            "model.diffusion_model.input_blocks.8.0.out_layers.3.weight": "blocks.19.conv2.weight",
            "model.diffusion_model.input_blocks.8.1.norm.bias": "blocks.20.norm.bias",
            "model.diffusion_model.input_blocks.8.1.norm.weight": "blocks.20.norm.weight",
            "model.diffusion_model.input_blocks.8.1.proj_in.bias": "blocks.20.proj_in.bias",
            "model.diffusion_model.input_blocks.8.1.proj_in.weight": "blocks.20.proj_in.weight",
            "model.diffusion_model.input_blocks.8.1.proj_out.bias": "blocks.20.proj_out.bias",
            "model.diffusion_model.input_blocks.8.1.proj_out.weight": "blocks.20.proj_out.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "blocks.20.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.20.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.20.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "blocks.20.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "blocks.20.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "blocks.20.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.20.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.20.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "blocks.20.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "blocks.20.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.20.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.20.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "blocks.20.transformer_blocks.0.ff.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "blocks.20.transformer_blocks.0.ff.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias": "blocks.20.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight": "blocks.20.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias": "blocks.20.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight": "blocks.20.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias": "blocks.20.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight": "blocks.20.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.input_blocks.9.0.op.bias": "blocks.22.conv.bias",
            "model.diffusion_model.input_blocks.9.0.op.weight": "blocks.22.conv.weight",
            "model.diffusion_model.middle_block.0.emb_layers.1.bias": "blocks.28.time_emb_proj.bias",
            "model.diffusion_model.middle_block.0.emb_layers.1.weight": "blocks.28.time_emb_proj.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.bias": "blocks.28.norm1.bias",
            "model.diffusion_model.middle_block.0.in_layers.0.weight": "blocks.28.norm1.weight",
            "model.diffusion_model.middle_block.0.in_layers.2.bias": "blocks.28.conv1.bias",
            "model.diffusion_model.middle_block.0.in_layers.2.weight": "blocks.28.conv1.weight",
            "model.diffusion_model.middle_block.0.out_layers.0.bias": "blocks.28.norm2.bias",
            "model.diffusion_model.middle_block.0.out_layers.0.weight": "blocks.28.norm2.weight",
            "model.diffusion_model.middle_block.0.out_layers.3.bias": "blocks.28.conv2.bias",
            "model.diffusion_model.middle_block.0.out_layers.3.weight": "blocks.28.conv2.weight",
            "model.diffusion_model.middle_block.1.norm.bias": "blocks.29.norm.bias",
            "model.diffusion_model.middle_block.1.norm.weight": "blocks.29.norm.weight",
            "model.diffusion_model.middle_block.1.proj_in.bias": "blocks.29.proj_in.bias",
            "model.diffusion_model.middle_block.1.proj_in.weight": "blocks.29.proj_in.weight",
            "model.diffusion_model.middle_block.1.proj_out.bias": "blocks.29.proj_out.bias",
            "model.diffusion_model.middle_block.1.proj_out.weight": "blocks.29.proj_out.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight": "blocks.29.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.29.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.29.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": "blocks.29.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight": "blocks.29.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": "blocks.29.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.29.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.29.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight": "blocks.29.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": "blocks.29.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.29.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.29.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias": "blocks.29.transformer_blocks.0.ff.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight": "blocks.29.transformer_blocks.0.ff.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias": "blocks.29.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight": "blocks.29.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias": "blocks.29.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight": "blocks.29.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias": "blocks.29.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight": "blocks.29.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.middle_block.2.emb_layers.1.bias": "blocks.30.time_emb_proj.bias",
            "model.diffusion_model.middle_block.2.emb_layers.1.weight": "blocks.30.time_emb_proj.weight",
            "model.diffusion_model.middle_block.2.in_layers.0.bias": "blocks.30.norm1.bias",
            "model.diffusion_model.middle_block.2.in_layers.0.weight": "blocks.30.norm1.weight",
            "model.diffusion_model.middle_block.2.in_layers.2.bias": "blocks.30.conv1.bias",
            "model.diffusion_model.middle_block.2.in_layers.2.weight": "blocks.30.conv1.weight",
            "model.diffusion_model.middle_block.2.out_layers.0.bias": "blocks.30.norm2.bias",
            "model.diffusion_model.middle_block.2.out_layers.0.weight": "blocks.30.norm2.weight",
            "model.diffusion_model.middle_block.2.out_layers.3.bias": "blocks.30.conv2.bias",
            "model.diffusion_model.middle_block.2.out_layers.3.weight": "blocks.30.conv2.weight",
            "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
            "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
            "model.diffusion_model.out.2.bias": "conv_out.bias",
            "model.diffusion_model.out.2.weight": "conv_out.weight",
            "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias": "blocks.32.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight": "blocks.32.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.0.0.in_layers.0.bias": "blocks.32.norm1.bias",
            "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": "blocks.32.norm1.weight",
            "model.diffusion_model.output_blocks.0.0.in_layers.2.bias": "blocks.32.conv1.bias",
            "model.diffusion_model.output_blocks.0.0.in_layers.2.weight": "blocks.32.conv1.weight",
            "model.diffusion_model.output_blocks.0.0.out_layers.0.bias": "blocks.32.norm2.bias",
            "model.diffusion_model.output_blocks.0.0.out_layers.0.weight": "blocks.32.norm2.weight",
            "model.diffusion_model.output_blocks.0.0.out_layers.3.bias": "blocks.32.conv2.bias",
            "model.diffusion_model.output_blocks.0.0.out_layers.3.weight": "blocks.32.conv2.weight",
            "model.diffusion_model.output_blocks.0.0.skip_connection.bias": "blocks.32.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.0.0.skip_connection.weight": "blocks.32.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.1.0.emb_layers.1.bias": "blocks.34.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.1.0.emb_layers.1.weight": "blocks.34.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.1.0.in_layers.0.bias": "blocks.34.norm1.bias",
            "model.diffusion_model.output_blocks.1.0.in_layers.0.weight": "blocks.34.norm1.weight",
            "model.diffusion_model.output_blocks.1.0.in_layers.2.bias": "blocks.34.conv1.bias",
            "model.diffusion_model.output_blocks.1.0.in_layers.2.weight": "blocks.34.conv1.weight",
            "model.diffusion_model.output_blocks.1.0.out_layers.0.bias": "blocks.34.norm2.bias",
            "model.diffusion_model.output_blocks.1.0.out_layers.0.weight": "blocks.34.norm2.weight",
            "model.diffusion_model.output_blocks.1.0.out_layers.3.bias": "blocks.34.conv2.bias",
            "model.diffusion_model.output_blocks.1.0.out_layers.3.weight": "blocks.34.conv2.weight",
            "model.diffusion_model.output_blocks.1.0.skip_connection.bias": "blocks.34.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.1.0.skip_connection.weight": "blocks.34.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.10.0.emb_layers.1.bias": "blocks.62.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.10.0.emb_layers.1.weight": "blocks.62.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.10.0.in_layers.0.bias": "blocks.62.norm1.bias",
            "model.diffusion_model.output_blocks.10.0.in_layers.0.weight": "blocks.62.norm1.weight",
            "model.diffusion_model.output_blocks.10.0.in_layers.2.bias": "blocks.62.conv1.bias",
            "model.diffusion_model.output_blocks.10.0.in_layers.2.weight": "blocks.62.conv1.weight",
            "model.diffusion_model.output_blocks.10.0.out_layers.0.bias": "blocks.62.norm2.bias",
            "model.diffusion_model.output_blocks.10.0.out_layers.0.weight": "blocks.62.norm2.weight",
            "model.diffusion_model.output_blocks.10.0.out_layers.3.bias": "blocks.62.conv2.bias",
            "model.diffusion_model.output_blocks.10.0.out_layers.3.weight": "blocks.62.conv2.weight",
            "model.diffusion_model.output_blocks.10.0.skip_connection.bias": "blocks.62.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.10.0.skip_connection.weight": "blocks.62.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.10.1.norm.bias": "blocks.63.norm.bias",
            "model.diffusion_model.output_blocks.10.1.norm.weight": "blocks.63.norm.weight",
            "model.diffusion_model.output_blocks.10.1.proj_in.bias": "blocks.63.proj_in.bias",
            "model.diffusion_model.output_blocks.10.1.proj_in.weight": "blocks.63.proj_in.weight",
            "model.diffusion_model.output_blocks.10.1.proj_out.bias": "blocks.63.proj_out.bias",
            "model.diffusion_model.output_blocks.10.1.proj_out.weight": "blocks.63.proj_out.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight": "blocks.63.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.63.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.63.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight": "blocks.63.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight": "blocks.63.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight": "blocks.63.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.63.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.63.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight": "blocks.63.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight": "blocks.63.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.63.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.63.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias": "blocks.63.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight": "blocks.63.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias": "blocks.63.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight": "blocks.63.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias": "blocks.63.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight": "blocks.63.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias": "blocks.63.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight": "blocks.63.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.11.0.emb_layers.1.bias": "blocks.65.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.11.0.emb_layers.1.weight": "blocks.65.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.11.0.in_layers.0.bias": "blocks.65.norm1.bias",
            "model.diffusion_model.output_blocks.11.0.in_layers.0.weight": "blocks.65.norm1.weight",
            "model.diffusion_model.output_blocks.11.0.in_layers.2.bias": "blocks.65.conv1.bias",
            "model.diffusion_model.output_blocks.11.0.in_layers.2.weight": "blocks.65.conv1.weight",
            "model.diffusion_model.output_blocks.11.0.out_layers.0.bias": "blocks.65.norm2.bias",
            "model.diffusion_model.output_blocks.11.0.out_layers.0.weight": "blocks.65.norm2.weight",
            "model.diffusion_model.output_blocks.11.0.out_layers.3.bias": "blocks.65.conv2.bias",
            "model.diffusion_model.output_blocks.11.0.out_layers.3.weight": "blocks.65.conv2.weight",
            "model.diffusion_model.output_blocks.11.0.skip_connection.bias": "blocks.65.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.11.0.skip_connection.weight": "blocks.65.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.11.1.norm.bias": "blocks.66.norm.bias",
            "model.diffusion_model.output_blocks.11.1.norm.weight": "blocks.66.norm.weight",
            "model.diffusion_model.output_blocks.11.1.proj_in.bias": "blocks.66.proj_in.bias",
            "model.diffusion_model.output_blocks.11.1.proj_in.weight": "blocks.66.proj_in.weight",
            "model.diffusion_model.output_blocks.11.1.proj_out.bias": "blocks.66.proj_out.bias",
            "model.diffusion_model.output_blocks.11.1.proj_out.weight": "blocks.66.proj_out.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight": "blocks.66.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.66.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.66.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight": "blocks.66.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight": "blocks.66.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight": "blocks.66.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.66.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.66.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight": "blocks.66.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight": "blocks.66.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.66.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.66.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias": "blocks.66.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight": "blocks.66.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias": "blocks.66.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight": "blocks.66.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias": "blocks.66.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight": "blocks.66.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias": "blocks.66.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight": "blocks.66.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.2.0.emb_layers.1.bias": "blocks.36.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.2.0.emb_layers.1.weight": "blocks.36.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.2.0.in_layers.0.bias": "blocks.36.norm1.bias",
            "model.diffusion_model.output_blocks.2.0.in_layers.0.weight": "blocks.36.norm1.weight",
            "model.diffusion_model.output_blocks.2.0.in_layers.2.bias": "blocks.36.conv1.bias",
            "model.diffusion_model.output_blocks.2.0.in_layers.2.weight": "blocks.36.conv1.weight",
            "model.diffusion_model.output_blocks.2.0.out_layers.0.bias": "blocks.36.norm2.bias",
            "model.diffusion_model.output_blocks.2.0.out_layers.0.weight": "blocks.36.norm2.weight",
            "model.diffusion_model.output_blocks.2.0.out_layers.3.bias": "blocks.36.conv2.bias",
            "model.diffusion_model.output_blocks.2.0.out_layers.3.weight": "blocks.36.conv2.weight",
            "model.diffusion_model.output_blocks.2.0.skip_connection.bias": "blocks.36.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.2.0.skip_connection.weight": "blocks.36.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.2.1.conv.bias": "blocks.37.conv.bias",
            "model.diffusion_model.output_blocks.2.1.conv.weight": "blocks.37.conv.weight",
            "model.diffusion_model.output_blocks.3.0.emb_layers.1.bias": "blocks.39.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.3.0.emb_layers.1.weight": "blocks.39.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.3.0.in_layers.0.bias": "blocks.39.norm1.bias",
            "model.diffusion_model.output_blocks.3.0.in_layers.0.weight": "blocks.39.norm1.weight",
            "model.diffusion_model.output_blocks.3.0.in_layers.2.bias": "blocks.39.conv1.bias",
            "model.diffusion_model.output_blocks.3.0.in_layers.2.weight": "blocks.39.conv1.weight",
            "model.diffusion_model.output_blocks.3.0.out_layers.0.bias": "blocks.39.norm2.bias",
            "model.diffusion_model.output_blocks.3.0.out_layers.0.weight": "blocks.39.norm2.weight",
            "model.diffusion_model.output_blocks.3.0.out_layers.3.bias": "blocks.39.conv2.bias",
            "model.diffusion_model.output_blocks.3.0.out_layers.3.weight": "blocks.39.conv2.weight",
            "model.diffusion_model.output_blocks.3.0.skip_connection.bias": "blocks.39.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.3.0.skip_connection.weight": "blocks.39.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.3.1.norm.bias": "blocks.40.norm.bias",
            "model.diffusion_model.output_blocks.3.1.norm.weight": "blocks.40.norm.weight",
            "model.diffusion_model.output_blocks.3.1.proj_in.bias": "blocks.40.proj_in.bias",
            "model.diffusion_model.output_blocks.3.1.proj_in.weight": "blocks.40.proj_in.weight",
            "model.diffusion_model.output_blocks.3.1.proj_out.bias": "blocks.40.proj_out.bias",
            "model.diffusion_model.output_blocks.3.1.proj_out.weight": "blocks.40.proj_out.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight": "blocks.40.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.40.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.40.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight": "blocks.40.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight": "blocks.40.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight": "blocks.40.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.40.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.40.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight": "blocks.40.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight": "blocks.40.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.40.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.40.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias": "blocks.40.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight": "blocks.40.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias": "blocks.40.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight": "blocks.40.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias": "blocks.40.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight": "blocks.40.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias": "blocks.40.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight": "blocks.40.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.4.0.emb_layers.1.bias": "blocks.42.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.4.0.emb_layers.1.weight": "blocks.42.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.4.0.in_layers.0.bias": "blocks.42.norm1.bias",
            "model.diffusion_model.output_blocks.4.0.in_layers.0.weight": "blocks.42.norm1.weight",
            "model.diffusion_model.output_blocks.4.0.in_layers.2.bias": "blocks.42.conv1.bias",
            "model.diffusion_model.output_blocks.4.0.in_layers.2.weight": "blocks.42.conv1.weight",
            "model.diffusion_model.output_blocks.4.0.out_layers.0.bias": "blocks.42.norm2.bias",
            "model.diffusion_model.output_blocks.4.0.out_layers.0.weight": "blocks.42.norm2.weight",
            "model.diffusion_model.output_blocks.4.0.out_layers.3.bias": "blocks.42.conv2.bias",
            "model.diffusion_model.output_blocks.4.0.out_layers.3.weight": "blocks.42.conv2.weight",
            "model.diffusion_model.output_blocks.4.0.skip_connection.bias": "blocks.42.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.4.0.skip_connection.weight": "blocks.42.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.4.1.norm.bias": "blocks.43.norm.bias",
            "model.diffusion_model.output_blocks.4.1.norm.weight": "blocks.43.norm.weight",
            "model.diffusion_model.output_blocks.4.1.proj_in.bias": "blocks.43.proj_in.bias",
            "model.diffusion_model.output_blocks.4.1.proj_in.weight": "blocks.43.proj_in.weight",
            "model.diffusion_model.output_blocks.4.1.proj_out.bias": "blocks.43.proj_out.bias",
            "model.diffusion_model.output_blocks.4.1.proj_out.weight": "blocks.43.proj_out.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "blocks.43.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.43.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.43.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "blocks.43.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "blocks.43.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "blocks.43.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.43.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.43.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "blocks.43.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "blocks.43.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.43.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.43.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "blocks.43.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "blocks.43.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias": "blocks.43.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight": "blocks.43.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias": "blocks.43.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight": "blocks.43.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias": "blocks.43.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight": "blocks.43.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.5.0.emb_layers.1.bias": "blocks.45.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.5.0.emb_layers.1.weight": "blocks.45.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.5.0.in_layers.0.bias": "blocks.45.norm1.bias",
            "model.diffusion_model.output_blocks.5.0.in_layers.0.weight": "blocks.45.norm1.weight",
            "model.diffusion_model.output_blocks.5.0.in_layers.2.bias": "blocks.45.conv1.bias",
            "model.diffusion_model.output_blocks.5.0.in_layers.2.weight": "blocks.45.conv1.weight",
            "model.diffusion_model.output_blocks.5.0.out_layers.0.bias": "blocks.45.norm2.bias",
            "model.diffusion_model.output_blocks.5.0.out_layers.0.weight": "blocks.45.norm2.weight",
            "model.diffusion_model.output_blocks.5.0.out_layers.3.bias": "blocks.45.conv2.bias",
            "model.diffusion_model.output_blocks.5.0.out_layers.3.weight": "blocks.45.conv2.weight",
            "model.diffusion_model.output_blocks.5.0.skip_connection.bias": "blocks.45.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.5.0.skip_connection.weight": "blocks.45.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.5.1.norm.bias": "blocks.46.norm.bias",
            "model.diffusion_model.output_blocks.5.1.norm.weight": "blocks.46.norm.weight",
            "model.diffusion_model.output_blocks.5.1.proj_in.bias": "blocks.46.proj_in.bias",
            "model.diffusion_model.output_blocks.5.1.proj_in.weight": "blocks.46.proj_in.weight",
            "model.diffusion_model.output_blocks.5.1.proj_out.bias": "blocks.46.proj_out.bias",
            "model.diffusion_model.output_blocks.5.1.proj_out.weight": "blocks.46.proj_out.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "blocks.46.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.46.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.46.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "blocks.46.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "blocks.46.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "blocks.46.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.46.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.46.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "blocks.46.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "blocks.46.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.46.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.46.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "blocks.46.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "blocks.46.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias": "blocks.46.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight": "blocks.46.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias": "blocks.46.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight": "blocks.46.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias": "blocks.46.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight": "blocks.46.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.5.2.conv.bias": "blocks.47.conv.bias",
            "model.diffusion_model.output_blocks.5.2.conv.weight": "blocks.47.conv.weight",
            "model.diffusion_model.output_blocks.6.0.emb_layers.1.bias": "blocks.49.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.6.0.emb_layers.1.weight": "blocks.49.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.6.0.in_layers.0.bias": "blocks.49.norm1.bias",
            "model.diffusion_model.output_blocks.6.0.in_layers.0.weight": "blocks.49.norm1.weight",
            "model.diffusion_model.output_blocks.6.0.in_layers.2.bias": "blocks.49.conv1.bias",
            "model.diffusion_model.output_blocks.6.0.in_layers.2.weight": "blocks.49.conv1.weight",
            "model.diffusion_model.output_blocks.6.0.out_layers.0.bias": "blocks.49.norm2.bias",
            "model.diffusion_model.output_blocks.6.0.out_layers.0.weight": "blocks.49.norm2.weight",
            "model.diffusion_model.output_blocks.6.0.out_layers.3.bias": "blocks.49.conv2.bias",
            "model.diffusion_model.output_blocks.6.0.out_layers.3.weight": "blocks.49.conv2.weight",
            "model.diffusion_model.output_blocks.6.0.skip_connection.bias": "blocks.49.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.6.0.skip_connection.weight": "blocks.49.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.6.1.norm.bias": "blocks.50.norm.bias",
            "model.diffusion_model.output_blocks.6.1.norm.weight": "blocks.50.norm.weight",
            "model.diffusion_model.output_blocks.6.1.proj_in.bias": "blocks.50.proj_in.bias",
            "model.diffusion_model.output_blocks.6.1.proj_in.weight": "blocks.50.proj_in.weight",
            "model.diffusion_model.output_blocks.6.1.proj_out.bias": "blocks.50.proj_out.bias",
            "model.diffusion_model.output_blocks.6.1.proj_out.weight": "blocks.50.proj_out.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight": "blocks.50.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.50.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.50.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight": "blocks.50.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight": "blocks.50.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight": "blocks.50.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.50.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.50.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight": "blocks.50.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight": "blocks.50.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.50.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.50.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias": "blocks.50.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight": "blocks.50.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias": "blocks.50.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight": "blocks.50.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias": "blocks.50.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight": "blocks.50.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias": "blocks.50.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight": "blocks.50.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.7.0.emb_layers.1.bias": "blocks.52.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.7.0.emb_layers.1.weight": "blocks.52.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.7.0.in_layers.0.bias": "blocks.52.norm1.bias",
            "model.diffusion_model.output_blocks.7.0.in_layers.0.weight": "blocks.52.norm1.weight",
            "model.diffusion_model.output_blocks.7.0.in_layers.2.bias": "blocks.52.conv1.bias",
            "model.diffusion_model.output_blocks.7.0.in_layers.2.weight": "blocks.52.conv1.weight",
            "model.diffusion_model.output_blocks.7.0.out_layers.0.bias": "blocks.52.norm2.bias",
            "model.diffusion_model.output_blocks.7.0.out_layers.0.weight": "blocks.52.norm2.weight",
            "model.diffusion_model.output_blocks.7.0.out_layers.3.bias": "blocks.52.conv2.bias",
            "model.diffusion_model.output_blocks.7.0.out_layers.3.weight": "blocks.52.conv2.weight",
            "model.diffusion_model.output_blocks.7.0.skip_connection.bias": "blocks.52.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.7.0.skip_connection.weight": "blocks.52.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.7.1.norm.bias": "blocks.53.norm.bias",
            "model.diffusion_model.output_blocks.7.1.norm.weight": "blocks.53.norm.weight",
            "model.diffusion_model.output_blocks.7.1.proj_in.bias": "blocks.53.proj_in.bias",
            "model.diffusion_model.output_blocks.7.1.proj_in.weight": "blocks.53.proj_in.weight",
            "model.diffusion_model.output_blocks.7.1.proj_out.bias": "blocks.53.proj_out.bias",
            "model.diffusion_model.output_blocks.7.1.proj_out.weight": "blocks.53.proj_out.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "blocks.53.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.53.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.53.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "blocks.53.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "blocks.53.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "blocks.53.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.53.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.53.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "blocks.53.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "blocks.53.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.53.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.53.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "blocks.53.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "blocks.53.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias": "blocks.53.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight": "blocks.53.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias": "blocks.53.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight": "blocks.53.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias": "blocks.53.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight": "blocks.53.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.8.0.emb_layers.1.bias": "blocks.55.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.8.0.emb_layers.1.weight": "blocks.55.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.8.0.in_layers.0.bias": "blocks.55.norm1.bias",
            "model.diffusion_model.output_blocks.8.0.in_layers.0.weight": "blocks.55.norm1.weight",
            "model.diffusion_model.output_blocks.8.0.in_layers.2.bias": "blocks.55.conv1.bias",
            "model.diffusion_model.output_blocks.8.0.in_layers.2.weight": "blocks.55.conv1.weight",
            "model.diffusion_model.output_blocks.8.0.out_layers.0.bias": "blocks.55.norm2.bias",
            "model.diffusion_model.output_blocks.8.0.out_layers.0.weight": "blocks.55.norm2.weight",
            "model.diffusion_model.output_blocks.8.0.out_layers.3.bias": "blocks.55.conv2.bias",
            "model.diffusion_model.output_blocks.8.0.out_layers.3.weight": "blocks.55.conv2.weight",
            "model.diffusion_model.output_blocks.8.0.skip_connection.bias": "blocks.55.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.8.0.skip_connection.weight": "blocks.55.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.8.1.norm.bias": "blocks.56.norm.bias",
            "model.diffusion_model.output_blocks.8.1.norm.weight": "blocks.56.norm.weight",
            "model.diffusion_model.output_blocks.8.1.proj_in.bias": "blocks.56.proj_in.bias",
            "model.diffusion_model.output_blocks.8.1.proj_in.weight": "blocks.56.proj_in.weight",
            "model.diffusion_model.output_blocks.8.1.proj_out.bias": "blocks.56.proj_out.bias",
            "model.diffusion_model.output_blocks.8.1.proj_out.weight": "blocks.56.proj_out.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "blocks.56.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.56.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.56.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "blocks.56.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "blocks.56.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "blocks.56.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.56.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.56.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "blocks.56.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "blocks.56.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.56.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.56.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "blocks.56.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "blocks.56.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias": "blocks.56.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight": "blocks.56.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias": "blocks.56.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight": "blocks.56.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias": "blocks.56.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight": "blocks.56.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.output_blocks.8.2.conv.bias": "blocks.57.conv.bias",
            "model.diffusion_model.output_blocks.8.2.conv.weight": "blocks.57.conv.weight",
            "model.diffusion_model.output_blocks.9.0.emb_layers.1.bias": "blocks.59.time_emb_proj.bias",
            "model.diffusion_model.output_blocks.9.0.emb_layers.1.weight": "blocks.59.time_emb_proj.weight",
            "model.diffusion_model.output_blocks.9.0.in_layers.0.bias": "blocks.59.norm1.bias",
            "model.diffusion_model.output_blocks.9.0.in_layers.0.weight": "blocks.59.norm1.weight",
            "model.diffusion_model.output_blocks.9.0.in_layers.2.bias": "blocks.59.conv1.bias",
            "model.diffusion_model.output_blocks.9.0.in_layers.2.weight": "blocks.59.conv1.weight",
            "model.diffusion_model.output_blocks.9.0.out_layers.0.bias": "blocks.59.norm2.bias",
            "model.diffusion_model.output_blocks.9.0.out_layers.0.weight": "blocks.59.norm2.weight",
            "model.diffusion_model.output_blocks.9.0.out_layers.3.bias": "blocks.59.conv2.bias",
            "model.diffusion_model.output_blocks.9.0.out_layers.3.weight": "blocks.59.conv2.weight",
            "model.diffusion_model.output_blocks.9.0.skip_connection.bias": "blocks.59.conv_shortcut.bias",
            "model.diffusion_model.output_blocks.9.0.skip_connection.weight": "blocks.59.conv_shortcut.weight",
            "model.diffusion_model.output_blocks.9.1.norm.bias": "blocks.60.norm.bias",
            "model.diffusion_model.output_blocks.9.1.norm.weight": "blocks.60.norm.weight",
            "model.diffusion_model.output_blocks.9.1.proj_in.bias": "blocks.60.proj_in.bias",
            "model.diffusion_model.output_blocks.9.1.proj_in.weight": "blocks.60.proj_in.weight",
            "model.diffusion_model.output_blocks.9.1.proj_out.bias": "blocks.60.proj_out.bias",
            "model.diffusion_model.output_blocks.9.1.proj_out.weight": "blocks.60.proj_out.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight": "blocks.60.transformer_blocks.0.attn1.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias": "blocks.60.transformer_blocks.0.attn1.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight": "blocks.60.transformer_blocks.0.attn1.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight": "blocks.60.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight": "blocks.60.transformer_blocks.0.attn1.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": "blocks.60.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias": "blocks.60.transformer_blocks.0.attn2.to_out.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight": "blocks.60.transformer_blocks.0.attn2.to_out.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight": "blocks.60.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight": "blocks.60.transformer_blocks.0.attn2.to_v.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias": "blocks.60.transformer_blocks.0.act_fn.proj.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight": "blocks.60.transformer_blocks.0.act_fn.proj.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias": "blocks.60.transformer_blocks.0.ff.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight": "blocks.60.transformer_blocks.0.ff.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias": "blocks.60.transformer_blocks.0.norm1.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight": "blocks.60.transformer_blocks.0.norm1.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias": "blocks.60.transformer_blocks.0.norm2.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight": "blocks.60.transformer_blocks.0.norm2.weight",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias": "blocks.60.transformer_blocks.0.norm3.bias",
            "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight": "blocks.60.transformer_blocks.0.norm3.weight",
            "model.diffusion_model.time_embed.0.bias": "time_embedding.0.bias",
            "model.diffusion_model.time_embed.0.weight": "time_embedding.0.weight",
            "model.diffusion_model.time_embed.2.bias": "time_embedding.2.bias",
            "model.diffusion_model.time_embed.2.weight": "time_embedding.2.weight",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if ".proj_in." in name or ".proj_out." in name:
                    param = param.squeeze()
                state_dict_[rename_dict[name]] = param
        return state_dict_