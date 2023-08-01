import torch
import modules.shared as shared
from compel import Compel, ReturnedEmbeddingsType

def compel_encode_prompt(pipeline, prompt, negative_prompt, prompt_2=None, negative_prompt_2=None, refiner=False):
    if "XL" not in pipeline.__class__.__name__:
      print(f"Compel parser is not configured for: {pipeline.__class__.__name__}")
      return None, None, None, None
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
    if not refiner:
      positive_te1 = compel_te1(prompt)
      positive_te2, pooled = compel_te2(prompt_2)
      positive = torch.cat((positive_te1, positive_te2), dim=-1)

      negative_te1 = compel_te1(negative_prompt)
      negative_te2, negative_pooled = compel_te2(negative_prompt_2)
      negative = torch.cat((negative_te1, negative_te2), dim=-1)
    if refiner:
      positive, pooled = compel_te2(prompt)
      negative, negative_pooled = compel_te2(negative_prompt)
    
    
    [prompt_embed, negative_embed] = compel_te2.pad_conditioning_tensors_to_same_length([positive, negative])
    return prompt_embed, pooled, negative_embed, negative_pooled