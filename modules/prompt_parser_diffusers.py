import torch
import modules.shared as shared
import modules.prompt_parser as prompt_parser
from compel import Compel, ReturnedEmbeddingsType
import diffusers
import typing

def convert_to_compel(prompt: str):
    if prompt is None:
      return None
    all_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt, 100)[0] #100 should be steps, but doesn't actually matter because we can't schedule yet
    output_list = prompt_parser.parse_prompt_attention(all_schedules[0][1])
    converted_prompt = []
    for subprompt, weight in output_list:
      if subprompt != " ":
        if weight == 1:
          converted_prompt.append(subprompt)
        else:
          converted_prompt.append(f"({subprompt}){weight}")
    converted_prompt = " ".join(converted_prompt)
    return converted_prompt

def compel_encode_prompt(pipeline: typing.Any, *args, **kwargs):
   compel_encode_fn = COMPEL_ENCODE_FN_DICT.get(type(pipeline), None)
   if compel_encode_fn is None:
      raise TypeError(f"Compel encoding not yet supported for {type(pipeline).__name__}.")
   return compel_encode_fn(pipeline, *args, **kwargs)

def compel_encode_prompt_sdxl(pipeline: diffusers.StableDiffusionXLPipeline, prompt: str, negative_prompt: str, prompt_2: typing.Optional[str]=None, negative_prompt_2: typing.Optional[str]=None, is_refiner: bool = False):
    if shared.opts.data['prompt_attention'] != 'Compel parser':
        prompt = convert_to_compel(prompt)
        negative_prompt = convert_to_compel(negative_prompt)
        prompt_2 = convert_to_compel(prompt_2)
        negative_prompt_2 = convert_to_compel(negative_prompt_2)

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
    if not is_refiner:
      positive_te1 = compel_te1(prompt)
      positive_te2, positive_pooled = compel_te2(prompt_2)
      positive = torch.cat((positive_te1, positive_te2), dim=-1)

      negative_te1 = compel_te1(negative_prompt)
      negative_te2, negative_pooled = compel_te2(negative_prompt_2)
      negative = torch.cat((negative_te1, negative_te2), dim=-1)
    else:
      positive, positive_pooled = compel_te2(prompt)
      negative, negative_pooled = compel_te2(negative_prompt)

    shared.log.debug(compel_te1.parse_prompt_string(prompt))
    [prompt_embed, negative_embed] = compel_te2.pad_conditioning_tensors_to_same_length([positive, negative])
    return prompt_embed, positive_pooled, negative_embed, negative_pooled

COMPEL_ENCODE_FN_DICT = {diffusers.StableDiffusionXLPipeline: compel_encode_prompt_sdxl}
