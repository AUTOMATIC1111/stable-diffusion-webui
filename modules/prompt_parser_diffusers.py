import os
import typing
import torch
import diffusers
from compel import Compel, ReturnedEmbeddingsType
import modules.shared as shared
import modules.prompt_parser as prompt_parser

debug_output = os.environ.get('SD_PROMPT_DEBUG', None)
debug = shared.log.info if debug_output is not None else lambda *args, **kwargs: None


def convert_to_compel(prompt: str):
    if prompt is None:
        return None
    all_schedules = prompt_parser.get_learned_conditioning_prompt_schedules([prompt], 100)[0]
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


CLIP_SKIP_MAPPING = {
    None: ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    1: ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    2: ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
}


def compel_encode_prompts(
    pipeline: diffusers.StableDiffusionXLPipeline | diffusers.StableDiffusionPipeline,
    prompts: list,
    negative_prompts: list,
    prompts_2: typing.Optional[list] = None,
    negative_prompts_2: typing.Optional[list] = None,
    is_refiner: bool = None,
    clip_skip: typing.Optional[int] = None,
):
    prompt_embeds = []
    positive_pooleds = []
    negative_embeds = []
    negative_pooleds = []
    for i in range(len(prompts)):
        prompt_embed, positive_pooled, negative_embed, negative_pooled = compel_encode_prompt(pipeline,
                                                                                              prompts[i],
                                                                                              negative_prompts[i],
                                                                                              prompts_2[i] if prompts_2 is not None else None,
                                                                                              negative_prompts_2[i] if negative_prompts_2 is not None else None,
                                                                                              is_refiner, clip_skip)
        prompt_embeds.append(prompt_embed)
        positive_pooleds.append(positive_pooled)
        negative_embeds.append(negative_embed)
        negative_pooleds.append(negative_pooled)

    prompt_embeds = torch.cat(prompt_embeds, dim=0)
    negative_embeds = torch.cat(negative_embeds, dim=0)
    if shared.sd_model_type == "sdxl":
        positive_pooleds = torch.cat(positive_pooleds, dim=0)
        negative_pooleds = torch.cat(negative_pooleds, dim=0)
    return prompt_embeds, positive_pooleds, negative_embeds, negative_pooleds


def compel_encode_prompt(
    pipeline: diffusers.StableDiffusionXLPipeline | diffusers.StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    prompt_2: typing.Optional[str] = None,
    negative_prompt_2: typing.Optional[str] = None,
    is_refiner: bool = None,
    clip_skip: typing.Optional[int] = None,
):
    if shared.sd_model_type not in {"sd", "sdxl"}:
        shared.log.warning(f"Prompt parser: Compel not supported: {type(pipeline).__name__}")
        return (None, None, None, None)

    if shared.sd_model_type == "sdxl":
        embedding_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        if clip_skip is not None and clip_skip > 1:
            shared.log.warning(f"Prompt parser SDXL unsupported: clip_skip={clip_skip}")
    else:
        embedding_type = CLIP_SKIP_MAPPING.get(clip_skip, ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED)
        if clip_skip not in CLIP_SKIP_MAPPING:
            shared.log.warning(f"Prompt parser unsupported: clip_skip={clip_skip} expected={set(CLIP_SKIP_MAPPING.keys())}")

    if shared.opts.data["prompt_attention"] != "Compel parser":
        prompt = convert_to_compel(prompt)
        negative_prompt = convert_to_compel(negative_prompt)
        prompt_2 = convert_to_compel(prompt_2)
        negative_prompt_2 = convert_to_compel(negative_prompt_2)

    compel_te1 = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        returned_embeddings_type=embedding_type,
        requires_pooled=False,
        # truncate_long_prompts=False,
        device=shared.device
    )

    if shared.sd_model_type == "sdxl":
        compel_te2 = Compel(
            tokenizer=pipeline.tokenizer_2,
            text_encoder=pipeline.text_encoder_2,
            returned_embeddings_type=embedding_type,
            requires_pooled=True,
            device=shared.device
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
        parsed = compel_te1.parse_prompt_string(prompt)
        debug(f"Prompt parser Compel: {parsed}")
        [prompt_embed, negative_embed] = compel_te2.pad_conditioning_tensors_to_same_length([positive, negative])
        return prompt_embed, positive_pooled, negative_embed, negative_pooled

    positive, negative = compel_te1(prompt), compel_te1(negative_prompt)
    [prompt_embed, negative_embed] = compel_te1.pad_conditioning_tensors_to_same_length([positive, negative])
    return prompt_embed, None, negative_embed, None
