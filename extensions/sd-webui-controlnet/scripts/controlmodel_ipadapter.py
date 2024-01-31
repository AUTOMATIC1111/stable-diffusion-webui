import math
from typing import List, NamedTuple, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from transformers.models.clip.modeling_clip import CLIPVisionModelOutput
from scripts.logging import logger


class ImageEmbed(NamedTuple):
    """Image embed for a single image."""
    cond_emb: torch.Tensor
    uncond_emb: torch.Tensor

    def eval(self, cond_mark: torch.Tensor) -> torch.Tensor:
        assert cond_mark.ndim == 4
        assert self.cond_emb.ndim == self.uncond_emb.ndim == 3
        assert self.cond_emb.shape[0] == self.uncond_emb.shape[0] == 1
        cond_mark = cond_mark[:, :, :, 0].to(self.cond_emb)
        device = cond_mark.device
        dtype = cond_mark.dtype
        return (
            self.cond_emb.to(device=device, dtype=dtype) * cond_mark +
            self.uncond_emb.to(device=device, dtype=dtype) * (1 - cond_mark)
        )

    def average_of(*args: List[Tuple[torch.Tensor, torch.Tensor]]) -> "ImageEmbed":
        conds, unconds = zip(*args)
        def average_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.sum(torch.stack(tensors), dim=0) / len(tensors)
        return ImageEmbed(average_tensors(conds), average_tensors(unconds))


class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class MLPProjModelFaceId(torch.nn.Module):
    """ MLPProjModel used for FaceId.
    Source: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter_faceid.py
    """
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        clip_extra_context_tokens = self.proj(id_embeds)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(-1, self.num_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens



class FacePerceiverResampler(torch.nn.Module):
    """ Source: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter_faceid.py """
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class ProjModelFaceIdPlus(torch.nn.Module):
    """ Source: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter_faceid.py """
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, scale=1.0, shortcut=False):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# Cross Attention to_k, to_v for IPAdapter
class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class IPAdapterModel(torch.nn.Module):
    def __init__(self, state_dict, clip_embeddings_dim, cross_attention_dim,
                 is_plus, sdxl_plus, is_full, is_faceid: bool, is_portrait: bool,
                 is_instantid: bool):
        super().__init__()
        self.device = "cpu"

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.is_plus = is_plus
        self.sdxl_plus = sdxl_plus
        self.is_full = is_full
        self.clip_extra_context_tokens = 16 if (self.is_plus or is_portrait) else 4

        if is_instantid:
            self.image_proj_model = self.init_proj_instantid()
        elif is_faceid:
            self.image_proj_model = self.init_proj_faceid()
        elif self.is_plus:
            if self.is_full:
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=cross_attention_dim,
                    clip_embeddings_dim=clip_embeddings_dim
                )
            else:
                self.image_proj_model = Resampler(
                    dim=1280 if sdxl_plus else cross_attention_dim,
                    depth=4,
                    dim_head=64,
                    heads=20 if sdxl_plus else 12,
                    num_queries=self.clip_extra_context_tokens,
                    embedding_dim=clip_embeddings_dim,
                    output_dim=self.cross_attention_dim,
                    ff_mult=4
                )
        else:
            self.clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // self.cross_attention_dim

            self.image_proj_model = ImageProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=clip_embeddings_dim,
                clip_extra_context_tokens=self.clip_extra_context_tokens
            )

        self.load_ip_adapter(state_dict)

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=4,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    def init_proj_instantid(self, image_emb_dim=512, num_tokens=16):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        )
        return image_proj_model

    def load_ip_adapter(self, state_dict):
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = To_KV(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, clip_vision_output: CLIPVisionModelOutput) -> ImageEmbed:
        self.image_proj_model.cpu()

        if self.is_plus:
            from annotator.clipvision import clip_vision_h_uc, clip_vision_vith_uc
            cond = self.image_proj_model(clip_vision_output['hidden_states'][-2].to(device='cpu', dtype=torch.float32))
            uncond = clip_vision_vith_uc.to(cond) if self.sdxl_plus else self.image_proj_model(clip_vision_h_uc.to(cond))
            return ImageEmbed(cond, uncond)

        clip_image_embeds = clip_vision_output['image_embeds'].to(device='cpu', dtype=torch.float32)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        # input zero vector for unconditional.
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return ImageEmbed(image_prompt_embeds, uncond_image_prompt_embeds)

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(
        self,
        face_embed: torch.Tensor,
        clip_vision_output: CLIPVisionModelOutput,
        is_v2: bool
    ) -> ImageEmbed:
        face_embed = face_embed.to(self.device, dtype=torch.float32)
        from annotator.clipvision import clip_vision_h_uc
        clip_embed = clip_vision_output['hidden_states'][-2].to(device=self.device, dtype=torch.float32)
        return ImageEmbed(
            self.image_proj_model(face_embed, clip_embed, shortcut=is_v2),
            self.image_proj_model(torch.zeros_like(face_embed), clip_vision_h_uc.to(clip_embed), shortcut=is_v2),
        )

    @torch.inference_mode()
    def get_image_embeds_faceid(self, insightface_output: torch.Tensor) -> ImageEmbed:
        """Get image embeds for non-plus faceid. Multiple inputs are supported."""
        self.image_proj_model.to(self.device)
        faceid_embed = insightface_output.to(self.device, dtype=torch.float32)
        return ImageEmbed(
            self.image_proj_model(faceid_embed),
            self.image_proj_model(torch.zeros_like(faceid_embed)),
        )

    @torch.inference_mode()
    def get_image_embeds_instantid(self, prompt_image_emb: Union[torch.Tensor, np.ndarray]) -> ImageEmbed:
        """Get image embeds for instantid."""
        image_proj_model_in_features = 512
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)

        prompt_image_emb = prompt_image_emb.to(device=self.device, dtype=torch.float32)
        prompt_image_emb = prompt_image_emb.reshape([1, -1, image_proj_model_in_features])
        return ImageEmbed(
            self.image_proj_model(prompt_image_emb),
            self.image_proj_model(torch.zeros_like(prompt_image_emb)),
        )


def get_block(model, flag):
    return {
        'input': model.input_blocks, 'middle': [model.middle_block], 'output': model.output_blocks
    }[flag]


def attn_forward_hacked(self, x, context=None, **kwargs):
    batch_size, sequence_length, inner_dim = x.shape
    h = self.heads
    head_dim = inner_dim // h

    if context is None:
        context = x

    q = self.to_q(x)
    k = self.to_k(context)
    v = self.to_v(context)

    del context

    q, k, v = map(
        lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

    del k, v

    for f in self.ipadapter_hacks:
        out = out + f(self, x, q)

    del q, x

    return self.to_out(out)


all_hacks = {}
current_model = None


def hack_blk(block, function, type):
    if not hasattr(block, 'ipadapter_hacks'):
        block.ipadapter_hacks = []

    if len(block.ipadapter_hacks) == 0:
        all_hacks[block] = block.forward
        block.forward = attn_forward_hacked.__get__(block, type)

    block.ipadapter_hacks.append(function)
    return


def set_model_attn2_replace(model, function, flag, id):
    from ldm.modules.attention import CrossAttention
    block = get_block(model, flag)[id][1].transformer_blocks[0].attn2
    hack_blk(block, function, CrossAttention)
    return


def set_model_patch_replace(model, function, flag, id, trans_id):
    from sgm.modules.attention import CrossAttention
    blk = get_block(model, flag)
    block = blk[id][1].transformer_blocks[trans_id].attn2
    hack_blk(block, function, CrossAttention)
    return


def clear_all_ip_adapter():
    global all_hacks, current_model
    for k, v in all_hacks.items():
        k.forward = v
        k.ipadapter_hacks = []
    all_hacks = {}
    current_model = None
    return


class PlugableIPAdapter(torch.nn.Module):
    def __init__(self, state_dict, model_name: str):
        """
        Arguments:
            - state_dict: model state_dict.
            - model_name: file name of the model.
        """
        super().__init__()
        self.is_v2 = "v2" in model_name
        self.is_faceid = "faceid" in model_name
        self.is_instantid = "instant_id" in model_name
        self.is_portrait = "portrait" in model_name
        self.is_full = "proj.3.weight" in state_dict['image_proj']
        self.is_plus = (
            self.is_full or
            "latents" in state_dict["image_proj"] or
            "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]
        )
        cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.sdxl = cross_attention_dim == 2048
        self.sdxl_plus = self.sdxl and self.is_plus
        if self.is_faceid and self.is_v2 and self.is_plus:
            logger.info("IP-Adapter faceid plus v2 detected.")

        if self.is_instantid:
            # InstantID does not use clip embedding.
            clip_embeddings_dim = None
        elif self.is_faceid:
            if self.is_plus:
                clip_embeddings_dim = 1280
            else:
                # Plain faceid does not use clip_embeddings_dim.
                clip_embeddings_dim = None
        elif self.is_plus:
            if self.sdxl_plus:
                clip_embeddings_dim = int(state_dict["image_proj"]["latents"].shape[2])
            elif self.is_full:
                clip_embeddings_dim = int(state_dict["image_proj"]["proj.0.weight"].shape[1])
            else:
                clip_embeddings_dim = int(state_dict['image_proj']['proj_in.weight'].shape[1])
        else:
            clip_embeddings_dim = int(state_dict['image_proj']['proj.weight'].shape[1])

        self.ipadapter = IPAdapterModel(state_dict,
                                        clip_embeddings_dim=clip_embeddings_dim,
                                        cross_attention_dim=cross_attention_dim,
                                        is_plus=self.is_plus,
                                        sdxl_plus=self.sdxl_plus,
                                        is_full=self.is_full,
                                        is_faceid=self.is_faceid,
                                        is_portrait=self.is_portrait,
                                        is_instantid=self.is_instantid)
        self.disable_memory_management = True
        self.dtype = None
        self.weight = 1.0
        self.cache = None
        self.p_start = 0.0
        self.p_end = 1.0
        return

    def reset(self):
        self.cache = {}
        return

    def get_image_emb(self, preprocessor_output) -> ImageEmbed:
        if self.is_instantid:
            return self.ipadapter.get_image_embeds_instantid(preprocessor_output)
        elif self.is_faceid and self.is_plus:
            # Note: FaceID plus uses both face_embed and clip_embed.
            # This should be the return value from preprocessor.
            return self.ipadapter.get_image_embeds_faceid_plus(
                preprocessor_output.face_embed,
                preprocessor_output.clip_embed,
                is_v2=self.is_v2
            )
        elif self.is_faceid:
            return self.ipadapter.get_image_embeds_faceid(preprocessor_output)
        else:
            return self.ipadapter.get_image_embeds(preprocessor_output)

    @torch.no_grad()
    def hook(self, model, preprocessor_outputs, weight, start, end, dtype=torch.float32):
        global current_model
        current_model = model

        self.p_start = start
        self.p_end = end

        self.cache = {}

        self.weight = weight
        device = torch.device('cpu')
        self.dtype = dtype

        self.ipadapter.to(device, dtype=self.dtype)
        if isinstance(preprocessor_outputs, (list, tuple)):
            preprocessor_outputs = preprocessor_outputs
        else:
            preprocessor_outputs = [preprocessor_outputs]
        self.image_emb = ImageEmbed.average_of(*[self.get_image_emb(o) for o in preprocessor_outputs])
        # From https://github.com/laksjdjf/IPAdapter-ComfyUI
        if not self.sdxl:
            number = 0  # index of to_kvs
            for id in [1, 2, 4, 5, 7, 8]:  # id of input_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "input", id)
                number += 1
            for id in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # id of output_blocks that have cross attention
                set_model_attn2_replace(model, self.patch_forward(number), "output", id)
                number += 1
            set_model_attn2_replace(model, self.patch_forward(number), "middle", 0)
        else:
            number = 0
            for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, self.patch_forward(number), "input", id, index)
                    number += 1
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(model, self.patch_forward(number), "output", id, index)
                    number += 1
            for index in range(10):
                set_model_patch_replace(model, self.patch_forward(number), "middle", 0, index)
                number += 1

        return

    def call_ip(self, key: str, feat, device):
        if key in self.cache:
            return self.cache[key]
        else:
            ip = self.ipadapter.ip_layers.to_kvs[key](feat).to(device)
            self.cache[key] = ip
            return ip

    @torch.no_grad()
    def patch_forward(self, number: int):
        @torch.no_grad()
        def forward(attn_blk, x, q):
            batch_size, sequence_length, inner_dim = x.shape
            h = attn_blk.heads
            head_dim = inner_dim // h

            current_sampling_percent = getattr(current_model, 'current_sampling_percent', 0.5)
            if current_sampling_percent < self.p_start or current_sampling_percent > self.p_end:
                return 0

            k_key = f"{number * 2 + 1}_to_k_ip"
            v_key = f"{number * 2 + 1}_to_v_ip"
            cond_uncond_image_emb = self.image_emb.eval(current_model.cond_mark)
            ip_k = self.call_ip(k_key, cond_uncond_image_emb, device=q.device)
            ip_v = self.call_ip(v_key, cond_uncond_image_emb, device=q.device)

            ip_k, ip_v = map(
                lambda t: t.view(batch_size, -1, h, head_dim).transpose(1, 2),
                (ip_k, ip_v),
            )
            assert ip_k.dtype == ip_v.dtype

            # On MacOS, q can be float16 instead of float32.
            # https://github.com/Mikubill/sd-webui-controlnet/issues/2208
            if q.dtype != ip_k.dtype:
                ip_k = ip_k.to(dtype=q.dtype)
                ip_v = ip_v.to(dtype=q.dtype)

            ip_out = torch.nn.functional.scaled_dot_product_attention(q, ip_k, ip_v, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, h * head_dim)

            return ip_out * self.weight
        return forward
