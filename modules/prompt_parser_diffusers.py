import torch
import modules.shared as shared
from compel import Compel, ReturnedEmbeddingsType
import diffusers
import typing

def compel_encode_prompt(pipeline: typing.Any, *args, **kwargs):
   compel_encode_fn = COMPEL_ENCODE_FN_DICT.get(type(pipeline), None)
   if compel_encode_fn is None:
      raise TypeError(f"Compel encoding not yet supported for {type(pipeline).__name__}.")
   return compel_encode_fn(pipeline, *args, **kwargs)

def compel_encode_prompt_sdxl(pipeline: diffusers.StableDiffusionXLPipeline, prompt: str, negative_prompt: str, prompt_2: typing.Optional[str]=None, negative_prompt_2: typing.Optional[str]=None, refiner=False):
    compel_te1 = Compel(
      tokenizer=pipeline.tokenizer,
      text_encoder=pipeline.text_encoder,
      returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
      requires_pooled=False,
      )

    compel_te2 = Compel(
      tokenizer=pipeline.tokenizer_2,
      text_encoder=pipeline.text_encoder_2,
      returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
      requires_pooled=True,
      )
    if refiner is None:
      positive_te1 = compel_te1(prompt)
      positive_te2, pooled = compel_te2(prompt_2)
      positive = torch.cat((positive_te1, positive_te2), dim=-1)

      negative_te1 = compel_te1(negative_prompt)
      negative_te2, negative_pooled = compel_te2(negative_prompt_2)
      negative = torch.cat((negative_te1, negative_te2), dim=-1)
    else:
      positive, pooled = compel_te2(prompt)
      negative, negative_pooled = compel_te2(negative_prompt)
    
    
    [prompt_embed, negative_embed] = compel_te2.pad_conditioning_tensors_to_same_length([positive, negative])
    return prompt_embed, pooled, negative_embed, negative_pooled

COMPEL_ENCODE_FN_DICT = {diffusers.StableDiffusionXLPipeline: compel_encode_prompt_sdxl}