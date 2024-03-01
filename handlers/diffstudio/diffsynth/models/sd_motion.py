from .sd_unet import SDUNet, Attention, GEGLU
import torch
from einops import rearrange, repeat


class TemporalTransformerBlock(torch.nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, max_position_embeddings=32):
        super().__init__()

        # 1. Self-Attn
        self.pe1 = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, dim))
        self.norm1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = Attention(q_dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 2. Cross-Attn
        self.pe2 = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, dim))
        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = Attention(q_dim=dim, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True)

        # 3. Feed-forward
        self.norm3 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.act_fn = GEGLU(dim, dim * 4)
        self.ff = torch.nn.Linear(dim * 4, dim)


    def forward(self, hidden_states, batch_size=1):

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = rearrange(norm_hidden_states, "(b f) h c -> (b h) f c", b=batch_size)
        attn_output = self.attn1(norm_hidden_states + self.pe1[:, :norm_hidden_states.shape[1]])
        attn_output = rearrange(attn_output, "(b h) f c -> (b f) h c", b=batch_size)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = rearrange(norm_hidden_states, "(b f) h c -> (b h) f c", b=batch_size)
        attn_output = self.attn2(norm_hidden_states + self.pe2[:, :norm_hidden_states.shape[1]])
        attn_output = rearrange(attn_output, "(b h) f c -> (b f) h c", b=batch_size)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.act_fn(norm_hidden_states)
        ff_output = self.ff(ff_output)
        hidden_states = ff_output + hidden_states

        return hidden_states


class TemporalBlock(torch.nn.Module):
    
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, norm_num_groups=32, eps=1e-5):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            TemporalTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim
            )
            for d in range(num_layers)
        ])

        self.proj_out = torch.nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, time_emb, text_emb, res_stack, batch_size=1):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                batch_size=batch_size
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class SDMotionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_modules = torch.nn.ModuleList([
            TemporalBlock(8, 40, 320, eps=1e-6),
            TemporalBlock(8, 40, 320, eps=1e-6),
            TemporalBlock(8, 80, 640, eps=1e-6),
            TemporalBlock(8, 80, 640, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 160, 1280, eps=1e-6),
            TemporalBlock(8, 80, 640, eps=1e-6),
            TemporalBlock(8, 80, 640, eps=1e-6),
            TemporalBlock(8, 80, 640, eps=1e-6),
            TemporalBlock(8, 40, 320, eps=1e-6),
            TemporalBlock(8, 40, 320, eps=1e-6),
            TemporalBlock(8, 40, 320, eps=1e-6),
        ])
        self.call_block_id = {
            1: 0,
            4: 1,
            9: 2,
            12: 3,
            17: 4,
            20: 5,
            24: 6,
            26: 7,
            29: 8,
            32: 9,
            34: 10,
            36: 11,
            40: 12,
            43: 13,
            46: 14,
            50: 15,
            53: 16,
            56: 17,
            60: 18,
            63: 19,
            66: 20
        }
        
    def forward(self):
        pass

    def state_dict_converter(self):
        return SDMotionModelStateDictConverter()


class SDMotionModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "norm": "norm",
            "proj_in": "proj_in",
            "transformer_blocks.0.attention_blocks.0.to_q": "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.0.attention_blocks.0.to_k": "transformer_blocks.0.attn1.to_k",
            "transformer_blocks.0.attention_blocks.0.to_v": "transformer_blocks.0.attn1.to_v",
            "transformer_blocks.0.attention_blocks.0.to_out.0": "transformer_blocks.0.attn1.to_out",
            "transformer_blocks.0.attention_blocks.0.pos_encoder": "transformer_blocks.0.pe1",
            "transformer_blocks.0.attention_blocks.1.to_q": "transformer_blocks.0.attn2.to_q",
            "transformer_blocks.0.attention_blocks.1.to_k": "transformer_blocks.0.attn2.to_k",
            "transformer_blocks.0.attention_blocks.1.to_v": "transformer_blocks.0.attn2.to_v",
            "transformer_blocks.0.attention_blocks.1.to_out.0": "transformer_blocks.0.attn2.to_out",
            "transformer_blocks.0.attention_blocks.1.pos_encoder": "transformer_blocks.0.pe2",
            "transformer_blocks.0.norms.0": "transformer_blocks.0.norm1",
            "transformer_blocks.0.norms.1": "transformer_blocks.0.norm2",
            "transformer_blocks.0.ff.net.0.proj": "transformer_blocks.0.act_fn.proj",
            "transformer_blocks.0.ff.net.2": "transformer_blocks.0.ff",
            "transformer_blocks.0.ff_norm": "transformer_blocks.0.norm3",
            "proj_out": "proj_out",
        }
        name_list = sorted([i for i in state_dict if i.startswith("down_blocks.")])
        name_list += sorted([i for i in state_dict if i.startswith("mid_block.")])
        name_list += sorted([i for i in state_dict if i.startswith("up_blocks.")])
        state_dict_ = {}
        last_prefix, module_id = "", -1
        for name in name_list:
            names = name.split(".")
            prefix_index = names.index("temporal_transformer") + 1
            prefix = ".".join(names[:prefix_index])
            if prefix != last_prefix:
                last_prefix = prefix
                module_id += 1
            middle_name = ".".join(names[prefix_index:-1])
            suffix = names[-1]
            if "pos_encoder" in names:
                rename = ".".join(["motion_modules", str(module_id), rename_dict[middle_name]])
            else:
                rename = ".".join(["motion_modules", str(module_id), rename_dict[middle_name], suffix])
            state_dict_[rename] = state_dict[name]
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
