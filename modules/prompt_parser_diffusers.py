import os
import typing
import torch
from compel import ReturnedEmbeddingsType
from compel.embeddings_provider import BaseTextualInversionManager, EmbeddingsProvider
from modules import shared, prompt_parser


debug = shared.log.info if os.environ.get('SD_PROMPT_DEBUG', None) is not None else lambda *args, **kwargs: None


CLIP_SKIP_MAPPING = {
    None: ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    1: ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
    2: ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
}


# from https://github.com/damian0815/compel/blob/main/src/compel/diffusers_textual_inversion_manager.py
class DiffusersTextualInversionManager(BaseTextualInversionManager):
    def __init__(self, pipe, tokenizer):
        self.pipe = pipe
        self.tokenizer = tokenizer
        if hasattr(self.pipe, 'embedding_db'):
            self.pipe.embedding_db.embeddings_used.clear()

    # from https://github.com/huggingface/diffusers/blob/705c592ea98ba4e288d837b9cba2767623c78603/src/diffusers/loaders.py#L599
    def maybe_convert_prompt(self, prompt: typing.Union[str, typing.List[str]], tokenizer="PreTrainedTokenizer"):
        prompts = [prompt] if not isinstance(prompt, typing.List) else prompt
        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]
        if not isinstance(prompt, typing.List):
            return prompts[0]
        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer="PreTrainedTokenizer"):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                if hasattr(self.pipe, 'embedding_db'):
                    self.pipe.embedding_db.embeddings_used.append(token)
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1
                prompt = prompt.replace(token, replacement)
        if hasattr(self.pipe, 'embedding_db'):
            self.pipe.embedding_db.embeddings_used = list(set(self.pipe.embedding_db.embeddings_used))
        debug(f'Prompt: convert={prompt}')
        return prompt

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: typing.List[int]) -> typing.List[int]:
        if len(token_ids) == 0:
            return token_ids
        prompt = self.pipe.tokenizer.decode(token_ids)
        prompt = self.maybe_convert_prompt(prompt, self.pipe.tokenizer)
        debug(f'Prompt: expand={prompt}')
        return self.pipe.tokenizer.encode(prompt, add_special_tokens=False)


def encode_prompts(pipeline, prompts: list, negative_prompts: list, clip_skip: typing.Optional[int] = None):
    if 'StableDiffusion' not in pipeline.__class__.__name__:
        shared.log.warning(f"Prompt parser not supported: {pipeline.__class__.__name__}")
        return None, None, None, None
    else:
        prompt_embeds = []
        positive_pooleds = []
        negative_embeds = []
        negative_pooleds = []
        for i in range(len(prompts)):
            prompt_embed, positive_pooled, negative_embed, negative_pooled = get_weighted_text_embeddings_sdxl(pipeline,prompts[i], negative_prompts[i], clip_skip)
            prompt_embeds.append(prompt_embed)
            positive_pooleds.append(positive_pooled)
            negative_embeds.append(negative_embed)
            negative_pooleds.append(negative_pooled)

    if prompt_embeds is not None:
        prompt_embeds = torch.cat(prompt_embeds, dim=0)
    if negative_embeds is not None:
        negative_embeds = torch.cat(negative_embeds, dim=0)
    if positive_pooleds is not None and shared.sd_model_type == "sdxl":
        positive_pooleds = torch.cat(positive_pooleds, dim=0)
    if negative_pooleds is not None and shared.sd_model_type == "sdxl":
        negative_pooleds = torch.cat(negative_pooleds, dim=0)
    return prompt_embeds, positive_pooleds, negative_embeds, negative_pooleds


def get_prompts_with_weights(prompt: str):
    manager = DiffusersTextualInversionManager(shared.sd_model, shared.sd_model.tokenizer or shared.sd_model.tokenizer_2)
    prompt = manager.maybe_convert_prompt(prompt, shared.sd_model.tokenizer or shared.sd_model.tokenizer_2)
    texts_and_weights = prompt_parser.parse_prompt_attention(prompt)
    texts = [t for t, w in texts_and_weights]
    text_weights = [w for t, w in texts_and_weights]
    debug(f'Prompt: weights={texts_and_weights}')
    return texts, text_weights


def prepare_embedding_providers(pipe, clip_skip):
    embeddings_providers = []
    if 'XL' in pipe.__class__.__name__:
        embedding_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
    else:
        if clip_skip > 2:
            shared.log.warning(f"Prompt parser unsupported: clip_skip={clip_skip}")
            clip_skip = 2
        embedding_type = CLIP_SKIP_MAPPING[clip_skip]
    if getattr(pipe, "tokenizer", None) is not None and getattr(pipe, "text_encoder", None) is not None:
        embedding = EmbeddingsProvider(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate=False, returned_embeddings_type=embedding_type, device=pipe.device)
        embeddings_providers.append(embedding)
    if getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "text_encoder_2", None) is not None:
        embedding = EmbeddingsProvider(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, truncate=False, returned_embeddings_type=embedding_type, device=pipe.device)
        embeddings_providers.append(embedding)
    return embeddings_providers

def pad_to_same_length(embeds):
    try: #SDXL
        empty_embed = shared.sd_model.encode_prompt("")
    except Exception: #SD1.5
        empty_embed = shared.sd_model.encode_prompt("",shared.sd_model.device, 1, False)

    empty_batched = torch.cat([empty_embed[0]] * embeds[0].shape[0])
    max_token_count = max([embed.shape[1] for embed in embeds])
    for i, embed in enumerate(embeds):
        while embed.shape[1] < max_token_count:
            embed = torch.cat([embed, empty_batched], dim=1)
            embeds[i] = embed
    return embeds

def get_weighted_text_embeddings_sdxl(pipe, prompt: str = "", neg_prompt: str = "", clip_skip: int = None):
    prompt_2 = prompt.split("TE2:")[-1]
    neg_prompt_2 = neg_prompt.split("TE2:")[-1]
    prompt = prompt.split("TE2:")[0]
    neg_prompt = neg_prompt.split("TE2:")[0]

    ps = [get_prompts_with_weights(p) for p in [prompt, prompt_2]]
    positives = [t for t, w in ps]
    positive_weights = [w for t, w in ps]
    ns = [get_prompts_with_weights(p) for p in [neg_prompt, neg_prompt_2]]
    negatives = [t for t, w in ns]
    negative_weights = [w for t, w in ns]
    if getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "tokenizer", None) is None:
        positives.pop(0)
        positive_weights.pop(0)
        negatives.pop(0)
        negative_weights.pop(0)

    embedding_providers = prepare_embedding_providers(pipe, clip_skip)
    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds =  None

    for i in range(len(embedding_providers)):
        embed, ptokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(text_batch=[positives[i]], fragment_weights_batch=[positive_weights[i]], device=pipe.device, should_return_tokens=True)
        prompt_embeds.append(embed)
        embed, ntokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(text_batch=[negatives[i]], fragment_weights_batch=[negative_weights[i]],device=pipe.device, should_return_tokens=True)
        negative_prompt_embeds.append(embed)

    if prompt_embeds[-1].shape[-1] > 768:
        if shared.opts.diffusers_pooled == "weighted":
            pooled_prompt_embeds = prompt_embeds[-1][
                        torch.arange(prompt_embeds[-1].shape[0], device=pipe.device),
                        (ptokens.to(dtype=torch.int, device=pipe.device) == 49407)
                        .int()
                        .argmax(dim=-1),
                    ]
            negative_pooled_prompt_embeds = negative_prompt_embeds[-1][
                        torch.arange(negative_prompt_embeds[-1].shape[0], device=pipe.device),
                        (ntokens.to(dtype=torch.int, device=pipe.device) == 49407)
                        .int()
                        .argmax(dim=-1),
                    ]
        else:
            pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[prompt_2], device=pipe.device) if prompt_embeds[-1].shape[-1] > 768 else None
            negative_pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[neg_prompt_2], device=pipe.device) if negative_prompt_embeds[-1].shape[-1] > 768 else None

    prompt_embeds = torch.cat(prompt_embeds, dim=-1) if len(prompt_embeds) > 1 else prompt_embeds[0]
    negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=-1) if len(negative_prompt_embeds) > 1 else negative_prompt_embeds[0]
    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
        [prompt_embeds, negative_prompt_embeds] = pad_to_same_length([prompt_embeds, negative_prompt_embeds])
    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds
