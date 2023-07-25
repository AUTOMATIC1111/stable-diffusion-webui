import torch
from typing import Optional
import transformers.models.clip.modeling_clip

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    min = torch.tensor(torch.finfo(dtype).min, device="cpu")
    mask = torch.full((tgt_len, tgt_len), min, device=device) # https://discord.com/channels/1101998836328697867/1127441997184122920
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def CLIPTextEmbeddings_forward(
    self: transformers.models.clip.modeling_clip.CLIPTextEmbeddings,
    input_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
) -> torch.Tensor:
    from modules.devices import dtype
    seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

    if position_ids is None:
        position_ids = self.position_ids[:, :seq_length]

    if inputs_embeds is None:
        inputs_embeds = self.token_embedding(input_ids).type(dtype) # Type correction.

    position_embeddings = self.position_embedding(position_ids)
    embeddings = inputs_embeds + position_embeddings

    return embeddings

transformers.models.clip.modeling_clip._make_causal_mask = _make_causal_mask
transformers.models.clip.modeling_clip.CLIPTextEmbeddings.forward = CLIPTextEmbeddings_forward
